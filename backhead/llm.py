"""OpenAI client construction helpers."""

from __future__ import annotations

from openai import OpenAI


def create_openai_client(base_url: str, api_key: str) -> OpenAI:
    """Return an official OpenAI client configured for *base_url* and *api_key*.

    The initial endpoint may be a local llama.cpp llama-server or any
    OpenAI-compatible service; the Agent class only knows it receives an
    OpenAI client.
    """
    return OpenAI(base_url=base_url, api_key=api_key)