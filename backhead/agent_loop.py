"""Agent loop: stateful Agent class and generic create_agent() constructor."""

from __future__ import annotations

import json
from typing import Any

# Path to the workspace guide injected as a system instruction each turn.
# The file lives inside the container; change this to relocate the guide.
AGENT_WORKSPACE_GUIDE = "/workspace/AGENT.md"

# Default system prompt used when constructing email agents.
# Backend security rules are defined here; they must not be editable by the model.
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant running inside PodHead. "
    f"At the start of each session read {AGENT_WORKSPACE_GUIDE} for workspace instructions. "
    "Follow the instructions found there for all tool use and file management."
)


def history_to_openai_messages(history_rows: list[dict]) -> list[dict]:
    """Convert database message rows to OpenAI chat-completion message dicts.

    Only ``incoming`` (user) and ``outgoing`` (assistant) directions are
    handled; other row types are silently skipped.
    """
    messages: list[dict] = []
    for row in history_rows:
        direction = row.get("direction")
        content = row.get("content", "")
        if direction == "incoming":
            messages.append({"role": "user", "content": content})
        elif direction == "outgoing":
            messages.append({"role": "assistant", "content": content})
    return messages


class Agent:
    """Stateful conversation context bound to one set of backend dependencies.

    Each instance owns its own ``conversation_history``.  Separate instances
    never share history.  The official OpenAI client is stored as
    ``self.openai_client`` so callers can inspect or replace it.
    """

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
        self._children_spawned: int = 0

    def run(self, prompt: str) -> str:
        """Append *prompt* as a user message and run the agent loop to completion.

        The loop:
        1. Appends the prompt as a user message.
        2. Calls ``self.openai_client.chat.completions.create(...)`` with the
           system prompt, current history, and available tool schemas.
        3. Appends the assistant response to ``self.conversation_history``.
        4. If the response contains tool calls, executes each handler and
           appends the results in valid OpenAI tool-result format.
        5. Repeats until the model returns a final text response.
        6. Returns the final response text.
        """
        self.conversation_history.append({"role": "user", "content": prompt})

        while True:
            messages = [
                {"role": "system", "content": self.system_prompt},
                *self.conversation_history,
            ]
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
            }
            if self.tools:
                kwargs["tools"] = self.tools
                kwargs["tool_choice"] = "auto"

            response = self.openai_client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            msg = choice.message

            # Build the assistant turn to persist.
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

            # No tool calls → final text response.
            if not msg.tool_calls:
                return msg.content or ""

            # Execute each tool call and append results.
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                handler = self.tool_handlers.get(tool_name)
                if handler is None:
                    result = f"Error: unknown tool '{tool_name}'"
                else:
                    try:
                        result = handler(args, self)
                    except Exception as exc:  # noqa: BLE001
                        result = f"Error in tool '{tool_name}': {exc}"
                self.conversation_history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(result),
                    }
                )


def create_agent(
    *,
    openai_client: Any,
    model: str,
    system_prompt: str,
    conversation_history: list[dict] | None = None,
    tools: list[dict],
    tool_handlers: dict[str, Any],
    container_runner: Any,
    depth: int = 0,
    max_depth: int = 2,
    max_children: int = 4,
) -> Agent:
    """Generic backend helper for constructing an Agent instance.

    Both ``main.py`` (email-triggered agents) and ``spawn_subagent`` (child
    agents) must create agents through this single function so that the same
    construction path is always used.

    The caller supplies all concrete runtime dependencies; ``create_agent``
    does not assume a shared client, model, or tool set.
    """
    return Agent(
        openai_client=openai_client,
        model=model,
        system_prompt=system_prompt,
        conversation_history=conversation_history,
        tools=tools,
        tool_handlers=tool_handlers,
        container_runner=container_runner,
        depth=depth,
        max_depth=max_depth,
        max_children=max_children,
    )