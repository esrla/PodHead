"""Backend tool: run_cli.

Exposes a CLI execution tool that runs commands inside the shared Podman
container.  The actual execution boundary is an injected ``container_runner``
callable, so tests can supply a fake without a running container.
"""

from __future__ import annotations

from typing import Any

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
    """Return ``(schema, handler)`` for the ``run_cli`` tool.

    ``container_runner`` is the injected callable that performs actual
    execution.  Its signature is ``container_runner(command: str) -> str``.
    Tests pass a fake; production code passes the real Podman runner.

    The handler signature is ``handler(args, calling_agent)`` to match the
    convention used by all backend tool handlers.
    """

    def handler(args: dict, calling_agent: Any) -> str:
        command = args.get("command", "")
        return container_runner(command)

    return CLI_TOOL_SCHEMA, handler
