"""Unit tests for Agent construction, history conversion, subagents, and tree."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from backhead.agent_loop import (
    Agent,
    HISTORY_SEPARATOR,
    build_execution_tree,
    messages_to_openai_messages,
    tool_error_result,
)
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


def test_spawn_subagent_schema_only_exposes_prompt():
    params = SPAWN_SUBAGENT_SCHEMA["function"]["parameters"]
    assert set(params["properties"].keys()) == {"prompt"}
    assert params["required"] == ["prompt"]
    assert params["additionalProperties"] is False


def test_no_execution_tree_when_no_tools_used():
    client, _ = _fake_client(_FakeResponse("just a reply"))
    agent = Agent(openai_client=client, model="m", system_prompt="s")
    assert agent.run("hi") == "just a reply"
    assert build_execution_tree(agent, "just a reply") is None


def test_execution_tree_included_with_required_heading_when_tools_used():
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
    assert reply.startswith("[System-generated execution tree]\nMain agent")
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


def test_execution_tree_omits_binary_data_and_truncates_long_result():
    client, _ = _fake_client(
        _FakeResponse(None, [_FakeToolCall("c1", "run_cli", {"command": "x"})]),
        _FakeResponse("done"),
    )

    def handler(args, agent):
        return {"ok": True, "blob": b"x" * 200, "output": "y" * 300}

    agent = Agent(
        openai_client=client,
        model="m",
        system_prompt="s",
        tools=[{"type": "function", "function": {"name": "run_cli"}}],
        tool_handlers={"run_cli": handler},
    )
    reply = agent.run("go")
    assert "[binary data omitted]" in reply
    assert "... [truncated]" in reply
