"""Backend tool: run_cli."""

from __future__ import annotations

from typing import Any

from backhead.agent_loop import tool_error_result

CLI_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "run_cli",
        "description": (
            "Execute a shell command inside the shared workspace container. "
            "Returns the combined stdout/stderr output."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute.",
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}


def create_cli_tool(container_runner: Any) -> tuple[dict, Any]:
    def handler(args: dict, calling_agent: Any) -> str | dict:
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            return tool_error_result(
                "tool_execution_error",
                "Missing or invalid parameter.",
                "Expected a non-empty string in args['command'].",
            )
        try:
            return {"ok": True, "output": container_runner(command)}
        except TimeoutError as exc:
            return tool_error_result("tool_execution_error", "Command timed out.", str(exc))
        except Exception as exc:  # noqa: BLE001
            return tool_error_result(
                getattr(exc, "error_type", "tool_execution_error"),
                "Command failed.",
                str(exc),
            )

    return CLI_TOOL_SCHEMA, handler
