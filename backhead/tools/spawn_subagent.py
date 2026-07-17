"""Backend tool: spawn_subagent.

Exposes a single ``spawn_subagent`` function to the model.  The model
provides only the child prompt; all other configuration is supplied by the
backend via ``create_spawn_subagent_tool()``.
"""

from __future__ import annotations

from typing import Any

from backhead.agent_loop import create_agent

SPAWN_SUBAGENT_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "spawn_subagent",
        "description": (
            "Spawn a fresh agent context to handle a self-contained subtask. "
            "Returns the subagent's final response."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The task for the subagent.",
                },
            },
            "required": ["prompt"],
            "additionalProperties": False,
        },
    },
}


def create_spawn_subagent_tool(
    *,
    openai_client: Any,
    model: str,
    system_prompt: str,
    tools: list[dict],
    tool_handlers: dict[str, Any],
    container_runner: Any,
    max_depth: int = 2,
    max_children: int = 4,
) -> tuple[dict, Any]:
    """Return ``(schema, handler)`` for the ``spawn_subagent`` tool.

    The returned handler accepts ``(args, calling_agent)`` where
    ``calling_agent`` is the parent ``Agent`` instance.  The model only
    ever sees ``{"prompt": "..."}`` in the tool schema.

    Child agents are created through the same ``create_agent()`` helper used
    by ``main.py``, with fresh conversation history and the configuration
    captured in this closure.
    """

    def handler(args: dict, calling_agent: Any) -> str:
        if calling_agent.depth >= calling_agent.max_depth:
            return "Error: maximum spawn depth reached."
        if calling_agent._children_spawned >= calling_agent.max_children:
            return "Error: maximum child count reached."
        calling_agent._children_spawned += 1

        child = create_agent(
            openai_client=openai_client,
            model=model,
            system_prompt=system_prompt,
            conversation_history=None,
            tools=tools,
            tool_handlers=tool_handlers,
            container_runner=container_runner,
            depth=calling_agent.depth + 1,
            max_depth=max_depth,
            max_children=max_children,
        )
        return child.run(args["prompt"])

    return SPAWN_SUBAGENT_SCHEMA, handler
