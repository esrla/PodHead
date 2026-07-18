"""Agent loop and conversation-history conversion helpers."""

from __future__ import annotations

import json
from typing import Any

AGENT_WORKSPACE_GUIDE = "/workspace/AGENT.md"
HISTORY_SEPARATOR = "\n\n---\n\n"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant running inside PodHead. "
    f"At the start of each session read {AGENT_WORKSPACE_GUIDE} for workspace instructions. "
    "Follow the instructions found there for all tool use and file management."
)


def tool_error_result(
    error_type: str,
    message: str,
    details: str | None = None,
) -> dict[str, Any]:
    """Return a structured tool-error payload for the agent."""
    return {
        "ok": False,
        "error": {
            "type": error_type,
            "message": message,
            "details": details or "",
        },
    }


def history_to_openai_messages(history_rows: list[dict]) -> list[dict]:
    """Convert ordered database rows to OpenAI messages.

    Consecutive messages with the same role are collapsed into one message and
    separated with ``\n\n---\n\n``.
    """
    messages: list[dict[str, str]] = []
    for row in history_rows:
        direction = row.get("direction")
        content = row.get("content", "")
        if direction == "incoming":
            role = "user"
        elif direction == "outgoing":
            role = "assistant"
        else:
            continue

        if messages and messages[-1]["role"] == role:
            previous = messages[-1].get("content") or ""
            messages[-1]["content"] = (
                f"{previous}{HISTORY_SEPARATOR}{content}" if previous else content
            )
            continue

        messages.append({"role": role, "content": content})
    return messages


class Agent:
    """Stateful conversation context bound to one set of backend dependencies."""

    def __init__(
        self,
        *,
        openai_client: Any,
        model: str,
        system_prompt: str,
        conversation_history: list[dict] | None = None,
        tools: list[dict] | None = None,
        tool_handlers: dict[str, Any] | None = None,
        container_runner: Any = None,
        depth: int = 0,
        max_depth: int = 2,
        max_children: int = 4,
    ) -> None:
        self.openai_client = openai_client
        self.model = model
        self.system_prompt = system_prompt
        self.conversation_history: list[dict] = list(conversation_history or [])
        self.tools: list[dict] = list(tools or [])
        self.tool_handlers: dict[str, Any] = dict(tool_handlers or {})
        self.container_runner = container_runner
        self.depth = depth
        self.max_depth = max_depth
        self.max_children = max_children
        self._children_spawned = 0

    def _append_tool_result(self, tool_call_id: str, result: Any) -> None:
        if isinstance(result, (dict, list)):
            content = json.dumps(result)
        else:
            content = str(result)
        self.conversation_history.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
            }
        )

    def run(self, prompt: str) -> str:
        """Append *prompt* as a user message and run the agent loop to completion."""
        self.conversation_history.append({"role": "user", "content": prompt})

        while True:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    *self.conversation_history,
                ],
            }
            if self.tools:
                kwargs["tools"] = self.tools
                kwargs["tool_choice"] = "auto"

            response = self.openai_client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            msg = choice.message

            assistant_turn: dict[str, Any] = {
                "role": "assistant",
                "content": msg.content,
            }
            if msg.tool_calls:
                assistant_turn["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            self.conversation_history.append(assistant_turn)

            if not msg.tool_calls:
                return msg.content or ""

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                handler = self.tool_handlers.get(tool_name)
                if handler is None:
                    self._append_tool_result(
                        tc.id,
                        tool_error_result(
                            "tool_execution_error",
                            f"Unknown tool '{tool_name}'.",
                            "No backend handler is registered for this tool.",
                        ),
                    )
                    continue

                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError as exc:
                    self._append_tool_result(
                        tc.id,
                        tool_error_result(
                            "tool_execution_error",
                            "Invalid JSON arguments.",
                            str(exc),
                        ),
                    )
                    continue

                try:
                    result = handler(args, self)
                except TimeoutError as exc:
                    result = tool_error_result(
                        "tool_execution_error",
                        "Tool timed out.",
                        str(exc),
                    )
                except Exception as exc:  # noqa: BLE001
                    error_type = getattr(exc, "error_type", "tool_execution_error")
                    result = tool_error_result(
                        error_type,
                        "Tool execution failed.",
                        str(exc),
                    )
                self._append_tool_result(tc.id, result)
