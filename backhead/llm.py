"""OpenAI client construction helpers."""

from __future__ import annotations

from openai import OpenAI


def create_openai_client(base_url: str, api_key: str) -> OpenAI:
    """Return an OpenAI-compatible client."""
    return OpenAI(base_url=base_url, api_key=api_key)


def test_openai_endpoint(openai_client: OpenAI, model: str) -> None:
    """Run a minimal request against a configured endpoint."""
    openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=1,
    )
