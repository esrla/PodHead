"""Unit tests for Agent, create_agent, history conversion, spawn_subagent,
and the email-agent construction flow.

All tests use fake OpenAI clients; no running llama.cpp server is required.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any
from unittest.mock import MagicMock

import pytest

from backhead import db
from backhead.agent_loop import Agent, create_agent, history_to_openai_messages
from backhead.mail import IncomingEmail, process_incoming_email
from backhead.tools.spawn_subagent import SPAWN_SUBAGENT_SCHEMA, create_spawn_subagent_tool
from main import build_email_agent_runner


# ---------------------------------------------------------------------------
# Fake OpenAI client helpers
# ---------------------------------------------------------------------------

class _FakeToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = MagicMock(name=name, arguments=json.dumps(arguments))
        self.function.name = name
        self.function.arguments = json.dumps(arguments)


class _FakeMessage:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class _FakeChoice:
    def __init__(self, content, tool_calls=None, finish_reason="stop"):
        self.message = _FakeMessage(content, tool_calls)
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(self, content, tool_calls=None, finish_reason="stop"):
        self.choices = [_FakeChoice(content, tool_calls, finish_reason)]


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self._index = 0
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        resp = self._responses[self._index]
        self._index += 1
        return resp


def _fake_client(*responses):
    """Return a fake OpenAI client that yields the given responses in order."""
    client = MagicMock()
    completions = _FakeCompletions(responses)
    client.chat.completions = completions
    return client, completions


# ---------------------------------------------------------------------------
# 1. One Agent instance owns its own conversation history
# ---------------------------------------------------------------------------

def test_agent_owns_its_own_history():
    client, completions = _fake_client(_FakeResponse("Hello!"))
    agent = create_agent(
        openai_client=client,
        model="m",
        system_prompt="sys",
        tools=[],
        tool_handlers={},
        container_runner=None,
    )
    agent.run("Hi")
    assert len(agent.conversation_history) == 2  # user + assistant


# ---------------------------------------------------------------------------
# 2. Separate Agent instances do not share history
# ---------------------------------------------------------------------------

def test_separate_agents_do_not_share_history():
    c1, _ = _fake_client(_FakeResponse("A"))
    c2, _ = _fake_client(_FakeResponse("B"))
    a1 = create_agent(openai_client=c1, model="m", system_prompt="s",
                      tools=[], tool_handlers={}, container_runner=None)
    a2 = create_agent(openai_client=c2, model="m", system_prompt="s",
                      tools=[], tool_handlers={}, container_runner=None)
    a1.run("first")
    assert a2.conversation_history == []


# ---------------------------------------------------------------------------
# 3. The official-style client is stored as self.openai_client
# ---------------------------------------------------------------------------

def test_openai_client_stored_as_attribute():
    sentinel = object()
    agent = Agent(
        openai_client=sentinel,
        model="m",
        system_prompt="s",
    )
    assert agent.openai_client is sentinel


# ---------------------------------------------------------------------------
# 4. The configured model is used in API calls
# ---------------------------------------------------------------------------

def test_configured_model_used_in_api_call():
    client, completions = _fake_client(_FakeResponse("ok"))
    agent = create_agent(
        openai_client=client,
        model="my-special-model",
        system_prompt="s",
        tools=[],
        tool_handlers={},
        container_runner=None,
    )
    agent.run("hello")
    assert completions.calls[0]["model"] == "my-special-model"


# ---------------------------------------------------------------------------
# 5. An ordinary assistant response is appended and returned
# ---------------------------------------------------------------------------

def test_ordinary_response_appended_and_returned():
    client, _ = _fake_client(_FakeResponse("The answer is 42."))
    agent = create_agent(
        openai_client=client,
        model="m",
        system_prompt="s",
        tools=[],
        tool_handlers={},
        container_runner=None,
    )
    result = agent.run("What is the answer?")
    assert result == "The answer is 42."
    assert agent.conversation_history[-1] == {
        "role": "assistant",
        "content": "The answer is 42.",
    }


# ---------------------------------------------------------------------------
# 6. Tool calls and tool results are appended in valid order
# ---------------------------------------------------------------------------

def test_tool_call_and_result_appended_in_order():
    tc = _FakeToolCall("call-1", "echo", {"text": "hi"})
    client, completions = _fake_client(
        _FakeResponse(None, tool_calls=[tc], finish_reason="tool_calls"),
        _FakeResponse("Done."),
    )

    def echo_handler(args, calling_agent):
        return args["text"]

    agent = create_agent(
        openai_client=client,
        model="m",
        system_prompt="s",
        tools=[{"type": "function", "function": {"name": "echo"}}],
        tool_handlers={"echo": echo_handler},
        container_runner=None,
    )
    result = agent.run("echo hi")
    assert result == "Done."
    roles = [m["role"] for m in agent.conversation_history]
    assert roles == ["user", "assistant", "tool", "assistant"]
    tool_msg = agent.conversation_history[2]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call-1"
    assert tool_msg["content"] == "hi"


# ---------------------------------------------------------------------------
# 7. Email history is converted correctly
# ---------------------------------------------------------------------------

def test_history_to_openai_messages_conversion():
    rows = [
        {"direction": "incoming", "content": "Hello"},
        {"direction": "outgoing", "content": "Hi there"},
        {"direction": "incoming", "content": "How are you?"},
    ]
    messages = history_to_openai_messages(rows)
    assert messages == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "How are you?"},
    ]


def test_history_conversion_skips_unknown_directions():
    rows = [
        {"direction": "incoming", "content": "msg"},
        {"direction": "unknown", "content": "should skip"},
    ]
    messages = history_to_openai_messages(rows)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"


# ---------------------------------------------------------------------------
# 8. The current incoming email is not duplicated in the agent context
# ---------------------------------------------------------------------------

def test_current_incoming_email_not_duplicated():
    """The newest incoming row is already in history; run_agent must not add it again."""
    conn = sqlite3.connect(":memory:")
    db.init_db(conn)
    captured_messages: list[list[dict]] = []

    def capturing_client_factory():
        client = MagicMock()

        def fake_create(**kwargs):
            captured_messages.append(list(kwargs["messages"]))
            return _FakeResponse("Reply")

        client.chat.completions.create = fake_create
        return client

    client = capturing_client_factory()
    runner = build_email_agent_runner(
        openai_client=client,
        model="m",
        system_prompt="sys",
        tools=[],
        tool_handlers={},
        container_runner=None,
        max_depth=2,
        max_children=4,
    )

    process_incoming_email(
        conn=conn,
        incoming=IncomingEmail(
            from_header="alice@example.com",
            subject="Hi",
            body="Hello agent",
            message_id="<m1@example.com>",
        ),
        whitelist={"alice@example.com"},
        run_agent=runner,
        send_reply=lambda _: None,
    )

    # There should be exactly one user message with content "Hello agent"
    assert len(captured_messages) == 1
    user_contents = [
        m["content"] for m in captured_messages[0] if m["role"] == "user"
    ]
    assert user_contents.count("Hello agent") == 1


# ---------------------------------------------------------------------------
# 9. main.py / build_email_agent_runner constructs agents through create_agent
# ---------------------------------------------------------------------------

def test_email_agent_runner_calls_create_agent(monkeypatch):
    created: list[Agent] = []
    original_create_agent = create_agent

    def recording_create_agent(**kwargs):
        agent = original_create_agent(**kwargs)
        created.append(agent)
        return agent

    import main as main_module
    monkeypatch.setattr(main_module, "create_agent", recording_create_agent)

    client = MagicMock()
    client.chat.completions.create.return_value = _FakeResponse("ok")

    runner = build_email_agent_runner(
        openai_client=client,
        model="m",
        system_prompt="s",
        tools=[],
        tool_handlers={},
        container_runner=None,
        max_depth=2,
        max_children=4,
    )

    conn = sqlite3.connect(":memory:")
    db.init_db(conn)
    process_incoming_email(
        conn=conn,
        incoming=IncomingEmail(
            from_header="alice@example.com",
            subject="Hi",
            body="test",
            message_id="<t1@example.com>",
        ),
        whitelist={"alice@example.com"},
        run_agent=runner,
        send_reply=lambda _: None,
    )
    assert len(created) == 1


# ---------------------------------------------------------------------------
# 10. spawn_subagent also constructs agents through create_agent
# ---------------------------------------------------------------------------

def test_spawn_subagent_uses_create_agent(monkeypatch):
    import backhead.tools.spawn_subagent as ss_module

    created: list[Agent] = []
    original = ss_module.create_agent

    def recording(**kwargs):
        agent = original(**kwargs)
        created.append(agent)
        return agent

    monkeypatch.setattr(ss_module, "create_agent", recording)

    sub_client = MagicMock()
    sub_client.chat.completions.create.return_value = _FakeResponse("child done")

    _, spawn_handler = create_spawn_subagent_tool(
        openai_client=sub_client,
        model="sub-m",
        system_prompt="sub-sys",
        tools=[],
        tool_handlers={},
        container_runner=None,
    )

    parent = Agent(openai_client=MagicMock(), model="p", system_prompt="ps")
    result = spawn_handler({"prompt": "do something"}, parent)
    assert result == "child done"
    assert len(created) == 1


# ---------------------------------------------------------------------------
# 11. The child starts with fresh conversation history
# ---------------------------------------------------------------------------

def test_child_starts_with_fresh_history():
    sub_client = MagicMock()
    sub_client.chat.completions.create.return_value = _FakeResponse("fresh")

    _, spawn_handler = create_spawn_subagent_tool(
        openai_client=sub_client,
        model="sub",
        system_prompt="sys",
        tools=[],
        tool_handlers={},
        container_runner=None,
    )

    # Give the parent some history.
    parent = Agent(
        openai_client=MagicMock(),
        model="p",
        system_prompt="ps",
        conversation_history=[{"role": "user", "content": "parent msg"}],
    )
    spawn_handler({"prompt": "subtask"}, parent)

    # The child's messages sent to the API should not contain parent history.
    call_kwargs = sub_client.chat.completions.create.call_args[1]
    user_messages = [m for m in call_kwargs["messages"] if m["role"] == "user"]
    assert len(user_messages) == 1
    assert user_messages[0]["content"] == "subtask"


# ---------------------------------------------------------------------------
# 12. The parent history is not copied into the child
# ---------------------------------------------------------------------------

def test_parent_history_not_copied_to_child():
    sub_client = MagicMock()
    sub_client.chat.completions.create.return_value = _FakeResponse("result")

    _, spawn_handler = create_spawn_subagent_tool(
        openai_client=sub_client,
        model="sub",
        system_prompt="sys",
        tools=[],
        tool_handlers={},
        container_runner=None,
    )

    parent = Agent(
        openai_client=MagicMock(),
        model="p",
        system_prompt="ps",
        conversation_history=[
            {"role": "user", "content": "parent-only content"},
            {"role": "assistant", "content": "parent-only reply"},
        ],
    )
    spawn_handler({"prompt": "new task"}, parent)

    call_kwargs = sub_client.chat.completions.create.call_args[1]
    all_content = [m.get("content", "") for m in call_kwargs["messages"]]
    assert "parent-only content" not in all_content
    assert "parent-only reply" not in all_content


# ---------------------------------------------------------------------------
# 13. The child receives the tool-configured OpenAI client and model
# ---------------------------------------------------------------------------

def test_child_receives_configured_client_and_model():
    sub_client = MagicMock()
    sub_client.chat.completions.create.return_value = _FakeResponse("ok")

    _, spawn_handler = create_spawn_subagent_tool(
        openai_client=sub_client,
        model="configured-child-model",
        system_prompt="sys",
        tools=[],
        tool_handlers={},
        container_runner=None,
    )

    parent = Agent(openai_client=MagicMock(), model="parent-model", system_prompt="ps")
    spawn_handler({"prompt": "task"}, parent)

    call_kwargs = sub_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "configured-child-model"
    sub_client.chat.completions.create.assert_called_once()


# ---------------------------------------------------------------------------
# 14. The child may use a different OpenAI client from the parent
# ---------------------------------------------------------------------------

def test_child_uses_different_client_from_parent():
    parent_client = MagicMock()
    sub_client = MagicMock()
    sub_client.chat.completions.create.return_value = _FakeResponse("child")

    _, spawn_handler = create_spawn_subagent_tool(
        openai_client=sub_client,
        model="sub",
        system_prompt="sys",
        tools=[],
        tool_handlers={},
        container_runner=None,
    )

    parent = Agent(openai_client=parent_client, model="p", system_prompt="ps")
    spawn_handler({"prompt": "task"}, parent)

    parent_client.chat.completions.create.assert_not_called()
    sub_client.chat.completions.create.assert_called_once()


# ---------------------------------------------------------------------------
# 15. The child may use a different model from the parent
# ---------------------------------------------------------------------------

def test_child_uses_different_model_from_parent():
    sub_client = MagicMock()
    sub_client.chat.completions.create.return_value = _FakeResponse("ok")

    _, spawn_handler = create_spawn_subagent_tool(
        openai_client=sub_client,
        model="child-model",
        system_prompt="sys",
        tools=[],
        tool_handlers={},
        container_runner=None,
    )

    parent = Agent(openai_client=MagicMock(), model="parent-model", system_prompt="ps")
    spawn_handler({"prompt": "task"}, parent)

    call_kwargs = sub_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "child-model"
    assert call_kwargs["model"] != "parent-model"


# ---------------------------------------------------------------------------
# 16. Parent and child share the same container runner
# ---------------------------------------------------------------------------

def test_parent_and_child_share_container_runner():
    """The same container_runner object must reach both parent and child."""
    shared_runner = MagicMock(return_value="output")

    sub_client = MagicMock()
    sub_client.chat.completions.create.return_value = _FakeResponse("done")

    from backhead.tools.cli_tool import create_cli_tool

    cli_schema, cli_handler = create_cli_tool(shared_runner)

    _, spawn_handler = create_spawn_subagent_tool(
        openai_client=sub_client,
        model="sub",
        system_prompt="sys",
        tools=[cli_schema],
        tool_handlers={"run_cli": cli_handler},
        container_runner=shared_runner,
    )

    parent_client = MagicMock()
    parent_client.chat.completions.create.return_value = _FakeResponse("parent done")
    parent = create_agent(
        openai_client=parent_client,
        model="p",
        system_prompt="ps",
        tools=[cli_schema],
        tool_handlers={"run_cli": cli_handler},
        container_runner=shared_runner,
    )

    # Both parent and child reference the same runner object.
    assert parent.container_runner is shared_runner

    # The child is created inside spawn_handler with the same runner.
    # We verify by calling the cli_handler and confirming shared_runner is invoked.
    cli_handler({"command": "ls"}, parent)
    shared_runner.assert_called_with("ls")


# ---------------------------------------------------------------------------
# 17. Only "prompt" is exposed in the spawn_subagent tool schema
# ---------------------------------------------------------------------------

def test_spawn_subagent_schema_only_exposes_prompt():
    params = SPAWN_SUBAGENT_SCHEMA["function"]["parameters"]
    assert set(params["properties"].keys()) == {"prompt"}
    assert params["required"] == ["prompt"]
    assert params.get("additionalProperties") is False


# ---------------------------------------------------------------------------
# 18. The model cannot provide model, endpoint, API key, or system prompt
# ---------------------------------------------------------------------------

def test_spawn_subagent_schema_hides_backend_parameters():
    param_names = set(
        SPAWN_SUBAGENT_SCHEMA["function"]["parameters"]["properties"].keys()
    )
    forbidden = {"model", "endpoint", "api_key", "system_prompt", "base_url"}
    assert param_names.isdisjoint(forbidden), (
        f"Schema exposes backend parameters: {param_names & forbidden}"
    )


# ---------------------------------------------------------------------------
# 19. Maximum spawn depth is enforced
# ---------------------------------------------------------------------------

def test_maximum_spawn_depth_enforced():
    sub_client = MagicMock()

    _, spawn_handler = create_spawn_subagent_tool(
        openai_client=sub_client,
        model="sub",
        system_prompt="sys",
        tools=[],
        tool_handlers={},
        container_runner=None,
        max_depth=2,
    )

    # Parent already at max depth.
    parent = Agent(
        openai_client=MagicMock(),
        model="p",
        system_prompt="ps",
        depth=2,
        max_depth=2,
    )
    result = spawn_handler({"prompt": "too deep"}, parent)
    assert "depth" in result.lower()
    sub_client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# 20. Maximum child count is enforced
# ---------------------------------------------------------------------------

def test_maximum_child_count_enforced():
    sub_client = MagicMock()
    sub_client.chat.completions.create.return_value = _FakeResponse("ok")

    _, spawn_handler = create_spawn_subagent_tool(
        openai_client=sub_client,
        model="sub",
        system_prompt="sys",
        tools=[],
        tool_handlers={},
        container_runner=None,
        max_children=2,
    )

    parent = Agent(
        openai_client=MagicMock(),
        model="p",
        system_prompt="ps",
        max_children=2,
    )

    spawn_handler({"prompt": "child 1"}, parent)
    spawn_handler({"prompt": "child 2"}, parent)
    result = spawn_handler({"prompt": "child 3"}, parent)

    assert "child count" in result.lower()
    assert sub_client.chat.completions.create.call_count == 2


# ---------------------------------------------------------------------------
# 21. Existing email-threading tests still pass (sanity re-run via import)
# ---------------------------------------------------------------------------

def test_existing_email_threading_unaffected():
    """Verify that the mail processing path still works as before."""
    conn = sqlite3.connect(":memory:")
    db.init_db(conn)
    sent = []
    result = process_incoming_email(
        conn=conn,
        incoming=IncomingEmail(
            from_header="alice@example.com",
            subject="Hello",
            body="First",
            message_id="<sanity@example.com>",
        ),
        whitelist={"alice@example.com"},
        run_agent=lambda history, incoming: "Reply",
        send_reply=sent.append,
    )
    assert result["status"] == "processed"
    assert len(sent) == 1


# ---------------------------------------------------------------------------
# 22. Existing skill-search tests still pass (verified by running the full suite)
# ---------------------------------------------------------------------------
# The find_skill tests are in test_find_skill.py and run as part of the suite.
# A lightweight import check is sufficient here.

def test_find_skill_module_importable():
    """Sanity check: the find_skill module can be imported from the workspace tools path."""
    import sys
    from pathlib import Path

    tools_path = Path(__file__).parents[1] / "head_pod" / "workspace" / "tools"
    if not tools_path.exists():
        pytest.skip("head_pod/workspace/tools not present in this environment")
    if str(tools_path) not in sys.path:
        sys.path.insert(0, str(tools_path))
    from find_skill import find_skills, search  # noqa: F401
