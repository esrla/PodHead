"""Backend tool: search_chat_history."""

from __future__ import annotations

from typing import Any, Callable

from backhead.agent_loop import tool_error_result

SEARCH_CHAT_HISTORY_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_chat_history",
        "description": (
            "Search semantically across the current sender's previous conversations. "
            "This searches backend-owned chat history and returns relevant earlier messages."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for in previous conversations.",
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Maximum number of results to return.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def create_search_chat_history_tool(search_fn: Callable[..., list[dict]] | None) -> tuple[dict, Any]:
    def handler(args: dict, calling_agent: Any) -> dict:
        query = args.get("query")
        max_results = args.get("max_results", 5)
        if not isinstance(query, str) or not query.strip():
            return tool_error_result(
                "tool_execution_error",
                "Missing or invalid parameter.",
                "Expected a non-empty string in args['query'].",
            )
        if not isinstance(max_results, int) or not 1 <= max_results <= 10:
            return tool_error_result(
                "tool_execution_error",
                "Missing or invalid parameter.",
                "Expected args['max_results'] to be an integer between 1 and 10.",
            )
        context = getattr(calling_agent, "backend_context", {}) or {}
        sender_id = context.get("sender_id")
        thread_id = context.get("thread_id")
        if not isinstance(sender_id, str) or not isinstance(thread_id, str):
            return tool_error_result(
                "tool_execution_error",
                "Chat history search is unavailable.",
                "Backend conversation metadata is missing.",
            )
        if search_fn is None:
            return tool_error_result(
                "tool_execution_error",
                "Chat history search is unavailable.",
                "Runtime search support is not configured.",
            )
        try:
            matches = search_fn(sender_id=sender_id, thread_id=thread_id, query=query, max_results=max_results)
        except Exception as exc:  # noqa: BLE001
            return tool_error_result(
                "tool_execution_error",
                "Chat history search failed.",
                str(exc),
            )
        return {"ok": True, "matches": matches}

    return SEARCH_CHAT_HISTORY_SCHEMA, handler
