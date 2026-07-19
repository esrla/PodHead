"""Unit tests for Agent construction, history conversion, subagents, and tree."""

from __future__ import annotations

import json
import sqlite3
from unittest.mock import MagicMock

from backhead import db
from backhead.agent_loop import (
    Agent,
    HISTORY_SEPARATOR,
    _redact_secrets,
    build_execution_tree,
    messages_to_openai_messages,
    tool_error_result,
)
from backhead.main import build_email_agent_runner
from backhead.tools.spawn_subagent import SPAWN_SUBAGENT_SCHEMA, create_spawn_subagent_tool


class _FakeToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = MagicMock()
        self.function.name = name
        self.function.arguments = arguments if isinstance(arguments, str) else json.dumps(arguments)


class _FakeMessage:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class _FakeChoice:
    def __init__(self, content, tool_calls=None):
        self.message = _FakeMessage(content, tool_calls)


class _FakeResponse:
    def __init__(self, content, tool_calls=None):
        self.choices = [_FakeChoice(content, tool_calls)]


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self._index = 0
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self._responses[self._index]
        self._index += 1
        return response


def _fake_client(*responses):
    client = MagicMock()
    completions = _FakeCompletions(responses)
    client.chat.completions = completions
    return client, completions


def _msg(role, *parts):
    return {"role": role, "content": [
        {"ordinal": i, "content_type": ct, "content": c} for i, (ct, c) in enumerate(parts)
    ]}


def test_tool_error_result_structure():
    assert tool_error_result("tool_execution_error", "Command failed.", "details") == {
        "ok": False,
        "error": {
            "type": "tool_execution_error",
            "message": "Command failed.",
            "details": "details",
        },
    }


def test_agent_is_constructed_directly_and_owns_its_history():
    client, _ = _fake_client(_FakeResponse("Hello"))
    agent = Agent(openai_client=client, model="main-model", system_prompt="sys")
    result = agent.run("Hi")
    assert result == "Hello"
    assert agent.conversation_history == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]


def test_messages_to_openai_collapses_consecutive_roles():
    conversation = [
        _msg("user", ("text", "one")),
        _msg("user", ("text", "two")),
        _msg("assistant", ("text", "three")),
        _msg("assistant", ("text", "four")),
    ]
    assert messages_to_openai_messages(conversation) == [
        {"role": "user", "content": f"one{HISTORY_SEPARATOR}two"},
        {"role": "assistant", "content": f"three{HISTORY_SEPARATOR}four"},
    ]


def test_messages_to_openai_image_without_media_root_uses_placeholder():
    conversation = [_msg("user", ("text", "look"), ("image", "media/x.png"))]
    result = messages_to_openai_messages(conversation)
    assert result[0]["role"] == "user"
    # All parts resolve to text (placeholder), so they collapse into a string.
    assert result[0]["content"] == "look[image: media/x.png]"


def test_messages_to_openai_image_with_media_root(tmp_path):
    from backhead import media
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 16
    rel = media.save_image(png, tmp_path)
    conversation = [_msg("user", ("image", rel))]
    result = messages_to_openai_messages(conversation, media_root=tmp_path)
    content = result[0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_main_agent_history_is_loaded_from_database_without_duplication():
    conn = sqlite3.connect(":memory:")
    db.init_db(conn)
    thread = db.create_email_thread(conn, "alice@example.com")

    def store(role, text, msg_id, state):
        mid = db.insert_message_with_content(
            conn, channel="email", thread_id=str(thread), sender_id="alice@example.com",
            role=role, timestamp=db.now_local_iso(), content_parts=[("text", text)],
        )
        db.insert_email_message_meta(
            conn, message_id=mid, email_message_id=msg_id, subject="Hello", process_state=state
        )
        return mid

    store("user", "First user message", "<old-in@example.com>", db.COMPLETED)
    store("assistant", "First assistant message", "<old-out@example.com>", db.COMPLETED)
    current_id = store("user", "Newest email", "<new-in@example.com>", db.PENDING)

    client, completions = _fake_client(_FakeResponse("Reply"))
    runner = build_email_agent_runner(
        openai_client=client,
        model="main-model",
        system_prompt="sys",
        tools=[],
        tool_handlers={},
        container_runner=None,
        max_depth=2,
        max_children=4,
    )
    history = db.get_conversation(conn, "email", str(thread))
    current_row = next(m for m in history if m["id"] == current_id)

    assert runner(history, current_row) == "Reply"
    messages = completions.calls[0]["messages"]
    user_contents = [message["content"] for message in messages if message["role"] == "user"]
    assert user_contents == ["First user message", "Newest email"]


def test_spawn_subagent_starts_fresh_history_and_uses_configured_model():
    sub_client, completions = _fake_client(_FakeResponse("child done"))
    _, handler = create_spawn_subagent_tool(
        openai_client=sub_client,
        model="sub-model",
        system_prompt="sub-system",
        tools=[],
        tool_handlers={},
        container_runner=None,
        max_depth=2,
        max_children=4,
    )
    parent = Agent(
        openai_client=MagicMock(),
        model="parent-model",
        system_prompt="parent-system",
        conversation_history=[{"role": "user", "content": "parent history"}],
    )

    result = handler({"prompt": "subtask"}, parent)
    assert result == {"ok": True, "response": "child done"}
    assert completions.calls[0]["model"] == "sub-model"
    assert [m["content"] for m in completions.calls[0]["messages"] if m["role"] == "user"] == ["subtask"]


def test_subagents_can_call_spawn_subagent_recursively():
    sub_client, completions = _fake_client(
        _FakeResponse(None, [_FakeToolCall("outer", "spawn_subagent", {"prompt": "inner"})]),
        _FakeResponse("inner complete"),
        _FakeResponse("outer complete"),
    )
    sub_tools: list[dict] = []
    sub_handlers: dict[str, object] = {}
    schema, handler = create_spawn_subagent_tool(
        openai_client=sub_client,
        model="sub-model",
        system_prompt="sub-system",
        tools=sub_tools,
        tool_handlers=sub_handlers,
        container_runner=None,
        max_depth=3,
        max_children=4,
    )
    sub_tools.append(schema)
    sub_handlers["spawn_subagent"] = handler

    parent = Agent(openai_client=MagicMock(), model="parent-model", system_prompt="parent-system")
    result = handler({"prompt": "outer"}, parent)
    assert result == {"ok": True, "response": "outer complete"}
    assert len(completions.calls) == 3


def test_spawn_limits_are_enforced():
    sub_client = MagicMock()
    _, handler = create_spawn_subagent_tool(
        openai_client=sub_client,
        model="sub-model",
        system_prompt="sub-system",
        tools=[],
        tool_handlers={},
        container_runner=None,
        max_depth=1,
        max_children=1,
    )

    depth_limited_parent = Agent(openai_client=MagicMock(), model="m", system_prompt="s", depth=1, max_depth=1)
    depth_result = handler({"prompt": "nope"}, depth_limited_parent)
    assert depth_result["ok"] is False
    assert "depth" in depth_result["error"]["message"].lower()

    child_limited_parent = Agent(openai_client=MagicMock(), model="m", system_prompt="s", max_children=1)
    child_limited_parent._children_spawned = 1
    child_result = handler({"prompt": "nope"}, child_limited_parent)
    assert child_result["ok"] is False
    assert "child count" in child_result["error"]["message"].lower()


def test_main_and_subagent_can_use_different_clients_and_models():
    main_client, main_completions = _fake_client(_FakeResponse("main done"))
    sub_client, sub_completions = _fake_client(_FakeResponse("sub done"))

    _, handler = create_spawn_subagent_tool(
        openai_client=sub_client,
        model="sub-model",
        system_prompt="sub-system",
        tools=[],
        tool_handlers={},
        container_runner=None,
    )
    parent = Agent(openai_client=main_client, model="main-model", system_prompt="main-system")

    assert parent.run("top level") == "main done"
    assert handler({"prompt": "delegate"}, parent) == {"ok": True, "response": "sub done"}
    assert main_completions.calls[0]["model"] == "main-model"
    assert sub_completions.calls[0]["model"] == "sub-model"


def test_tool_errors_are_returned_as_tool_messages_and_loop_continues():
    client, _ = _fake_client(
        _FakeResponse(None, [_FakeToolCall("call-1", "broken", "{not-json")]),
        _FakeResponse(None, [_FakeToolCall("call-2", "broken", json.dumps({}))]),
        _FakeResponse("recovered"),
    )

    def broken_handler(args, calling_agent):
        raise RuntimeError("should not run for invalid args")

    agent = Agent(
        openai_client=client,
        model="model",
        system_prompt="sys",
        tools=[{"type": "function", "function": {"name": "broken"}}],
        tool_handlers={"broken": broken_handler},
    )

    reply = agent.run("go")
    assert reply.endswith("recovered")
    tool_messages = [message for message in agent.conversation_history if message["role"] == "tool"]
    assert [message["tool_call_id"] for message in tool_messages] == ["call-1", "call-2"]
    first = json.loads(tool_messages[0]["content"])
    assert first["ok"] is False
    assert first["error"]["type"] == "tool_execution_error"


def test_spawn_subagent_schema_only_exposes_prompt():
    params = SPAWN_SUBAGENT_SCHEMA["function"]["parameters"]
    assert set(params["properties"].keys()) == {"prompt"}
    assert params["required"] == ["prompt"]
    assert params["additionalProperties"] is False


# --------------------------------------------------------------------------- #
# Execution tree + redaction
# --------------------------------------------------------------------------- #


def test_no_execution_tree_when_no_tools_used():
    client, _ = _fake_client(_FakeResponse("just a reply"))
    agent = Agent(openai_client=client, model="m", system_prompt="s")
    assert agent.run("hi") == "just a reply"
    assert build_execution_tree(agent, "just a reply") is None


def test_execution_tree_included_when_tools_used():
    client, _ = _fake_client(
        _FakeResponse(None, [_FakeToolCall("c1", "run_cli", {"command": "ls"})]),
        _FakeResponse("done"),
    )

    def handler(args, agent):
        return {"ok": True, "output": "file1\nfile2"}

    agent = Agent(
        openai_client=client,
        model="m",
        system_prompt="s",
        tools=[{"type": "function", "function": {"name": "run_cli"}}],
        tool_handlers={"run_cli": handler},
    )
    reply = agent.run("list files")
    assert "Main agent" in reply
    assert "Tool call: run_cli" in reply
    assert "Tool result:" in reply
    assert reply.endswith("\n---\ndone")


def test_execution_tree_shows_subagent_subtree():
    sub_client, _ = _fake_client(
        _FakeResponse(None, [_FakeToolCall("s1", "run_cli", {"command": "pwd"})]),
        _FakeResponse("child reply"),
    )
    sub_tools = [{"type": "function", "function": {"name": "run_cli"}}]

    def cli_handler(args, agent):
        return {"ok": True, "output": "/workspace"}

    sub_handlers = {"run_cli": cli_handler}
    _, spawn_handler = create_spawn_subagent_tool(
        openai_client=sub_client,
        model="sub-model",
        system_prompt="sub",
        tools=sub_tools,
        tool_handlers=sub_handlers,
        container_runner=None,
        max_depth=2,
        max_children=4,
    )

    main_client, _ = _fake_client(
        _FakeResponse(None, [_FakeToolCall("m1", "spawn_subagent", {"prompt": "do it"})]),
        _FakeResponse("all done"),
    )
    agent = Agent(
        openai_client=main_client,
        model="main",
        system_prompt="s",
        tools=[{"type": "function", "function": {"name": "spawn_subagent"}}],
        tool_handlers={"spawn_subagent": spawn_handler},
    )
    reply = agent.run("delegate")
    assert "Subagent" in reply
    assert "Tool call: spawn_subagent" in reply
    assert "Reply: child reply" in reply
    assert reply.endswith("\n---\nall done")


def test_secret_redaction_of_known_and_patterns():
    text = 'call with {"api_key": "sk-abcdef123456"} and token=abc'
    redacted = _redact_secrets(text, known_secrets=["sk-abcdef123456"])
    assert "sk-abcdef123456" not in redacted
    assert "[REDACTED]" in redacted


def test_secret_redaction_of_authorization_header():
    token = "supersecret" + "token"
    text = "Authorization: Bearer " + token
    redacted = _redact_secrets(text)
    assert token not in redacted
    assert "[REDACTED]" in redacted


def test_execution_tree_redacts_secrets_in_args():
    client, _ = _fake_client(
        _FakeResponse(None, [_FakeToolCall("c1", "run_cli", {"command": "login", "password": "hunter2000"})]),
        _FakeResponse("done"),
    )

    def handler(args, agent):
        return {"ok": True, "output": "ok"}

    agent = Agent(
        openai_client=client,
        model="m",
        system_prompt="s",
        tools=[{"type": "function", "function": {"name": "run_cli"}}],
        tool_handlers={"run_cli": handler},
        known_secrets=["hunter2000"],
    )
    reply = agent.run("go")
    assert "hunter2000" not in reply
    assert "[REDACTED]" in reply
