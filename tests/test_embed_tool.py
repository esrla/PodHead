# Tests for backhead.tools.embed_tool: schema, handler behaviour, and error paths.

from __future__ import annotations

import numpy as np
import pytest

from backhead.tools.embed_tool import EMBED_TOOL_SCHEMA, create_embed_tool


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_embed_fn(vec_size: int = 4):
    def embed_fn(texts):
        return np.array([[float(i) / 10 for i in range(vec_size)]] * len(texts))

    return embed_fn


# ── schema ─────────────────────────────────────────────────────────────────────


class TestEmbedToolSchema:
    def test_tool_name_is_embed_text(self):
        assert EMBED_TOOL_SCHEMA["function"]["name"] == "embed_text"

    def test_has_required_text_parameter(self):
        params = EMBED_TOOL_SCHEMA["function"]["parameters"]
        assert params["required"] == ["text"]
        assert "text" in params["properties"]

    def test_no_additional_properties_allowed(self):
        params = EMBED_TOOL_SCHEMA["function"]["parameters"]
        assert params["additionalProperties"] is False

    def test_type_is_function(self):
        assert EMBED_TOOL_SCHEMA["type"] == "function"


# ── handler ────────────────────────────────────────────────────────────────────


class TestEmbedToolHandler:
    def test_returns_ok_with_embedding_list(self):
        _, handler = create_embed_tool(_make_embed_fn(4))
        result = handler({"text": "hello world"}, None)
        assert result["ok"] is True
        assert "embedding" in result
        assert isinstance(result["embedding"], list)
        assert len(result["embedding"]) == 4

    def test_embedding_values_are_floats(self):
        _, handler = create_embed_tool(_make_embed_fn(3))
        result = handler({"text": "test"}, None)
        assert all(isinstance(v, float) for v in result["embedding"])

    def test_error_on_empty_string(self):
        _, handler = create_embed_tool(_make_embed_fn())
        result = handler({"text": ""}, None)
        assert result["ok"] is False
        assert result["error"]["type"] == "tool_execution_error"

    def test_error_on_whitespace_only_string(self):
        _, handler = create_embed_tool(_make_embed_fn())
        result = handler({"text": "   "}, None)
        assert result["ok"] is False

    def test_error_on_missing_text_key(self):
        _, handler = create_embed_tool(_make_embed_fn())
        result = handler({}, None)
        assert result["ok"] is False

    def test_error_on_non_string_text(self):
        _, handler = create_embed_tool(_make_embed_fn())
        result = handler({"text": 42}, None)
        assert result["ok"] is False

    def test_embed_fn_exception_returns_error(self):
        def bad_embed(texts):
            raise RuntimeError("model unavailable")

        _, handler = create_embed_tool(bad_embed)
        result = handler({"text": "hello"}, None)
        assert result["ok"] is False
        assert "Embedding failed" in result["error"]["message"]
        assert "model unavailable" in result["error"]["details"]

    def test_create_embed_tool_returns_schema_and_handler(self):
        schema, handler = create_embed_tool(_make_embed_fn())
        assert schema is EMBED_TOOL_SCHEMA
        assert callable(handler)
