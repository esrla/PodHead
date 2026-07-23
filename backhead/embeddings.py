"""OpenAI-compatible embeddings helpers for the trusted backend."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def create_openai_embed_fn(openai_client: Any, model: str) -> Callable[[list[str]], np.ndarray]:
    """Return an embed_fn that uses the configured embeddings API."""

    def embed(texts: list[str]) -> np.ndarray:
        response = openai_client.embeddings.create(model=model, input=texts)
        rows = sorted(response.data, key=lambda item: item.index)
        vectors = np.asarray([row.embedding for row in rows], dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError("Embedding API returned an unexpected shape.")
        return vectors

    return embed
