"""Backend tool: spawn_subagent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backhead.agent_loop import Agent, tool_error_result

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
    workspace_path: Path | None,
    skill_header_provider: Any,
    tools: list[dict],
    tool_handlers: dict[str, Any],
    container_runner: Any,
    max_depth: int = 2,
    max_children: int = 4,
) -> tuple[dict, Any]:
    def handler(args: dict, calling_agent: Any) -> str | dict:
        prompt = args.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return tool_error_result(
                "tool_execution_error",
                "Missing or invalid parameter.",
                "Expected a non-empty string in args['prompt'].",
            )
        if calling_agent.depth >= calling_agent.max_depth:
            return tool_error_result(
                "tool_execution_error",
                "Maximum spawn depth reached.",
                f"Current depth {calling_agent.depth} cannot exceed {calling_agent.max_depth}.",
            )
        if calling_agent._children_spawned >= calling_agent.max_children:
            return tool_error_result(
                "tool_execution_error",
                "Maximum child count reached.",
                f"Agent already spawned {calling_agent._children_spawned} children.",
            )

        calling_agent._children_spawned += 1
        child = Agent(
            openai_client=openai_client,
            model=model,
            system_prompt=system_prompt,
            workspace_path=workspace_path,
            skill_header_provider=skill_header_provider,
            tools=tools,
            tool_handlers=tool_handlers,
            container_runner=container_runner,
            depth=calling_agent.depth + 1,
            max_depth=max_depth,
            max_children=max_children,
            backend_context=calling_agent.backend_context,
        )
        try:
            response = child.run(prompt)
            calling_agent._pending_child_agent = child
            return {"ok": True, "response": response}
        except Exception as exc:  # noqa: BLE001
            return tool_error_result(
                getattr(exc, "error_type", "tool_execution_error"),
                "Subagent failed.",
                str(exc),
            )

    return SPAWN_SUBAGENT_SCHEMA, handler
