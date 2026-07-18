"""Unit tests for Agent construction, history conversion, and subagents."""

from __future__ import annotations

import json
import sqlite3
from unittest.mock import MagicMock

from backhead import db
from backhead.agent_loop import Agent, HISTORY_SEPARATOR, history_to_openai_messages, tool_error_result
from backhead.main import build_email_agent_runner
from backhead.tools.spawn_subagent import SPAWN_SUBAGENT_SCHEMA, create_spawn_subagent_tool


class _FakeToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = MagicMock()
        self.function.name = name
        self.function.arguments = arguments


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



def test_history_to_openai_messages_collapses_consecutive_roles():
    rows = [
        {"direction": "incoming", "content": "one"},
        {"direction": "incoming", "content": "two"},
        {"direction": "outgoing", "content": "three"},
        {"direction": "outgoing", "content": "four"},
    ]
    assert history_to_openai_messages(rows) == [
        {"role": "user", "content": f"one{HISTORY_SEPARATOR}two"},
        {"role": "assistant", "content": f"three{HISTORY_SEPARATOR}four"},
    ]



def test_main_agent_history_is_loaded_from_database_without_duplication():
    conn = sqlite3.connect(":memory:")
    db.init_db(conn)
    conversation_id = db.create_conversation(conn, "alice@example.com", created_ts=1)
    db.insert_message(
        conn,
        conversation_id=conversation_id,
        email_message_id="<old-in@example.com>",
        direction="incoming",
        content="First user message",
        subject="Hello",
        timestamp=1,
        process_state=db.COMPLETED,
    )
    db.insert_message(
        conn,
        conversation_id=conversation_id,
        email_message_id="<old-out@example.com>",
        direction="outgoing",
        content="First assistant message",
        subject="Re: Hello",
        timestamp=2,
        process_state=db.COMPLETED,
    )
    current_id = db.insert_message(
        conn,
        conversation_id=conversation_id,
        email_message_id="<new-in@example.com>",
        direction="incoming",
        content="Newest email",
        subject="Hello again",
        timestamp=3,
        process_state=db.PENDING,
    )

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
    history = db.get_conversation_history(conn, conversation_id)
    current_row = db.get_message(conn, current_id)

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
        _FakeResponse(None, [_FakeToolCall("outer", "spawn_subagent", json.dumps({"prompt": "inner"}))]),
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

    assert agent.run("go") == "recovered"
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
