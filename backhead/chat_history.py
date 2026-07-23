"""Backend-only chat history indexing, search, and compaction."""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Any, Callable

import numpy as np
import sqlite3

from backhead import db

COMPACTION_MIN_MESSAGES = 12
COMPACTION_KEEP_RECENT_MESSAGES = 6
COMPACTION_MAX_CHARS = 4000
SEARCH_RESULT_PREVIEW_CHARS = 280
DEFAULT_SEARCH_RESULTS = 5
MAX_SEARCH_RESULTS = 10


def _coerce_vector(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("Embedding must be a non-empty one-dimensional vector.")
    if not np.isfinite(vector).all():
        raise ValueError("Embedding must contain only finite values.")
    return vector


def vector_to_blob(vector: Any) -> bytes:
    return _coerce_vector(vector).astype(np.float32, copy=False).tobytes()


def blob_to_vector(blob: bytes) -> np.ndarray:
    vector = np.frombuffer(blob, dtype=np.float32)
    return _coerce_vector(vector)


def _message_part_payload(message: dict) -> list[dict[str, Any]]:
    return [
        {
            "ordinal": int(part["ordinal"]),
            "content_type": str(part["content_type"]),
            "content": str(part["content"]),
        }
        for part in sorted(message.get("content", []), key=lambda item: int(item["ordinal"]))
    ]


def _render_message_part(part: dict) -> str:
    if part["content_type"] == "text":
        return part["content"]
    return f"[{part['content_type']} content: {part['content']}]"


def build_searchable_message_text(message: dict) -> str:
    lines = [f"role: {message['role']}", f"timestamp: {message['timestamp']}", "", "parts:"]
    for part in _message_part_payload(message):
        lines.append(f"{part['ordinal']}: {_render_message_part(part)}")
    return "\n".join(lines)


def build_message_content_hash(message: dict, searchable_text: str, embedding_model: str) -> str:
    payload = {
        "embedding_model": embedding_model,
        "searchable_text_format": db.MESSAGE_SEARCHABLE_TEXT_FORMAT_VERSION,
        "message": {
            "id": int(message["id"]),
            "role": message["role"],
            "timestamp": message["timestamp"],
            "parts": _message_part_payload(message),
        },
        "searchable_text": searchable_text,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Embedding must have non-zero length.")
    return vector / norm


def _similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("Embedding dimensions do not match.")
    return float(np.dot(_normalize(left), _normalize(right)))


def _preview(text: str, max_chars: int = SEARCH_RESULT_PREVIEW_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _compaction_messages(conversation: list[dict]) -> list[dict]:
    if len(conversation) < COMPACTION_MIN_MESSAGES:
        return []
    older = conversation[:-COMPACTION_KEEP_RECENT_MESSAGES]
    return older if len(older) >= 2 else []


def build_compaction_hash(messages: list[dict]) -> str:
    payload = {
        "summary_format": db.COMPACTION_SUMMARY_FORMAT_VERSION,
        "messages": [
            {
                "id": int(message["id"]),
                "role": message["role"],
                "timestamp": message["timestamp"],
                "parts": _message_part_payload(message),
            }
            for message in messages
        ],
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_compaction_summary(messages: list[dict], *, max_chars: int = COMPACTION_MAX_CHARS) -> str:
    lines = [
        "[Compacted earlier conversation]",
        "These older messages stay unchanged in SQLite; this is only a backend-generated compact summary.",
    ]
    omitted = 0
    for message in messages:
        preview = _preview(build_searchable_message_text(message), max_chars=220).replace("\n", " | ")
        candidate = f"- id {message['id']} | {preview}"
        if len("\n".join(lines + [candidate])) > max_chars:
            omitted += 1
            continue
        lines.append(candidate)
    if omitted:
        lines.append(f"- omitted_messages: {omitted}")
    return "\n".join(lines)


def ensure_message_embeddings(
    conn: sqlite3.Connection,
    *,
    db_lock: threading.Lock,
    sender_id: str,
    document_embed_fn: Callable[[list[str]], np.ndarray],
    embedding_model: str,
) -> None:
    while True:
        with db_lock:
            candidate = db.find_oldest_message_needing_embedding(
                conn,
                sender_id=sender_id,
                embedding_model=embedding_model,
                searchable_text_builder=build_searchable_message_text,
                content_hash_builder=build_message_content_hash,
            )
        if candidate is None:
            return
        vector = document_embed_fn([candidate["searchable_text"]])[0]
        with db_lock:
            db.store_or_replace_message_embedding(
                conn,
                message_id=candidate["id"],
                embedding_model=embedding_model,
                content_hash=candidate["content_hash"],
                searchable_text=candidate["searchable_text"],
                embedding=vector_to_blob(vector),
                embedded_at=db.now_local_iso(),
            )


def search_previous_conversations(
    conn: sqlite3.Connection,
    *,
    db_lock: threading.Lock,
    sender_id: str,
    current_thread_id: str,
    query: str,
    query_embed_fn: Callable[[list[str]], np.ndarray],
    document_embed_fn: Callable[[list[str]], np.ndarray],
    embedding_model: str,
    limit: int = DEFAULT_SEARCH_RESULTS,
) -> list[dict]:
    ensure_message_embeddings(
        conn,
        db_lock=db_lock,
        sender_id=sender_id,
        document_embed_fn=document_embed_fn,
        embedding_model=embedding_model,
    )
    query_vector = query_embed_fn([query])[0]
    with db_lock:
        rows = db.get_message_embeddings_for_sender(
            conn,
            sender_id=sender_id,
            exclude_thread_id=current_thread_id,
        )
    scored: list[dict] = []
    for row in rows:
        try:
            score = _similarity(query_vector, blob_to_vector(row["embedding"]))
        except ValueError:
            continue
        scored.append(
            {
                "message_id": row["message_id"],
                "thread_id": row["thread_id"],
                "role": row["role"],
                "timestamp": row["timestamp"],
                "score": score,
                "preview": _preview(row["searchable_text"]),
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[: max(1, min(limit, MAX_SEARCH_RESULTS))]


def ensure_conversation_compaction(
    conn: sqlite3.Connection,
    *,
    db_lock: threading.Lock,
    channel: str,
    thread_id: str,
    sender_id: str,
    conversation: list[dict],
) -> dict | None:
    messages = _compaction_messages(conversation)
    if not messages:
        return None
    upto_message_id = int(messages[-1]["id"])
    content_hash = build_compaction_hash(messages)
    with db_lock:
        current = db.get_conversation_compaction(conn, channel=channel, thread_id=thread_id)
    if current and current["upto_message_id"] == upto_message_id and current["content_hash"] == content_hash:
        return current
    summary = build_compaction_summary(messages)
    with db_lock:
        db.store_or_replace_conversation_compaction(
            conn,
            channel=channel,
            thread_id=thread_id,
            sender_id=sender_id,
            upto_message_id=upto_message_id,
            content_hash=content_hash,
            summary=summary,
            summary_format=db.COMPACTION_SUMMARY_FORMAT_VERSION,
            compacted_at=db.now_local_iso(),
        )
        return db.get_conversation_compaction(conn, channel=channel, thread_id=thread_id)


def build_compacted_openai_history(
    conversation: list[dict],
    *,
    compaction: dict | None,
    recent_messages_to_openai: Callable[[list[dict]], list[dict]],
) -> list[dict]:
    if not compaction:
        return recent_messages_to_openai(conversation)
    recent_messages = [message for message in conversation if int(message["id"]) > int(compaction["upto_message_id"])]
    return [
        {"role": "system", "content": compaction["summary"]},
        *recent_messages_to_openai(recent_messages),
    ]
