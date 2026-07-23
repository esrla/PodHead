"""Backend tool: embed_text."""

from __future__ import annotations

from typing import Any, Callable

from backhead.agent_loop import tool_error_result

EMBED_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "embed_text",
        "description": (
            "Return a semantic embedding vector for the given text. "
            "Useful for computing similarity scores or comparing texts semantically."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to embed.",
                },
            },
            "required": ["text"],
            "additionalProperties": False,
        },
    },
}


def create_embed_tool(embed_fn: Callable) -> tuple[dict, Any]:
    """Return (schema, handler) for the embed_text tool.

    ``embed_fn(texts: list[str]) -> np.ndarray`` must return backend-generated
    embedding vectors from the configured embeddings API.
    """

    def handler(args: dict, calling_agent: Any) -> dict:
        text = args.get("text")
        if not isinstance(text, str) or not text.strip():
            return tool_error_result(
                "tool_execution_error",
                "Missing or invalid parameter.",
                "Expected a non-empty string in args['text'].",
            )
        try:
            vecs = embed_fn([text])
            return {"ok": True, "embedding": vecs[0].tolist()}
        except Exception as exc:  # noqa: BLE001
            return tool_error_result(
                "tool_execution_error",
                "Embedding failed.",
                str(exc),
            )

    return EMBED_TOOL_SCHEMA, handler
