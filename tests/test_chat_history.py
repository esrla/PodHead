from __future__ import annotations

import sqlite3
import threading

import numpy as np

from backhead import chat_history, db


def _conn():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    db.init_db(conn)
    return conn


def _make_embed(keyword: str):
    def embed_fn(texts):
        vectors = []
        for text in texts:
            vector = np.array([1.0, 0.0], dtype=np.float32) if keyword in text.lower() else np.array([0.0, 1.0], dtype=np.float32)
            vectors.append(vector)
        return np.array(vectors, dtype=np.float32)

    return embed_fn


def test_build_searchable_message_text_preserves_part_order_and_represents_non_text_parts():
    message = {
        "id": 1,
        "role": "user",
        "timestamp": "2026-07-23T12:30:00+02:00",
        "content": [
            {"ordinal": 0, "content_type": "text", "content": "first"},
            {"ordinal": 1, "content_type": "image", "content": "media/a.png"},
            {"ordinal": 2, "content_type": "text", "content": "last"},
        ],
    }

    searchable = chat_history.build_searchable_message_text(message)

    assert searchable == (
        "role: user\n"
        "timestamp: 2026-07-23T12:30:00+02:00\n\n"
        "parts:\n"
        "0: first\n"
        "1: [image content: media/a.png]\n"
        "2: last"
    )


def test_search_previous_conversations_uses_backend_embeddings_and_excludes_current_thread():
    conn = _conn()
    lock = threading.Lock()
    first = db.insert_message_with_content(
        conn,
        channel="email",
        thread_id="thread-old",
        sender_id="alice@example.com",
        role="user",
        timestamp="2026-01-01T00:00:00+00:00",
        content_parts=[("text", "project x status")],
    )
    db.insert_message_with_content(
        conn,
        channel="email",
        thread_id="thread-current",
        sender_id="alice@example.com",
        role="user",
        timestamp="2026-01-02T00:00:00+00:00",
        content_parts=[("text", "current thread mention")],
    )

    matches = chat_history.search_previous_conversations(
        conn,
        db_lock=lock,
        sender_id="alice@example.com",
        current_thread_id="thread-current",
        query="project x",
        query_embed_fn=_make_embed("project x"),
        document_embed_fn=_make_embed("project x"),
        embedding_model="embed-model",
        limit=5,
    )

    assert len(matches) == 1
    assert matches[0]["message_id"] == first
    assert matches[0]["thread_id"] == "thread-old"
    assert "project x status" in matches[0]["preview"]


def test_ensure_conversation_compaction_persists_side_data_without_rewriting_messages():
    conn = _conn()
    lock = threading.Lock()
    for index in range(12):
        db.insert_message_with_content(
            conn,
            channel="email",
            thread_id="thread-1",
            sender_id="alice@example.com",
            role="user" if index % 2 == 0 else "assistant",
            timestamp=f"2026-01-01T00:00:{index:02d}+00:00",
            content_parts=[("text", f"message {index}")],
        )
    conversation = db.get_conversation(conn, "email", "thread-1")

    compaction = chat_history.ensure_conversation_compaction(
        conn,
        db_lock=lock,
        channel="email",
        thread_id="thread-1",
        sender_id="alice@example.com",
        conversation=conversation,
    )

    unchanged = db.get_conversation(conn, "email", "thread-1")
    assert len(unchanged) == 12
    assert unchanged[0]["content"][0]["content"] == "message 0"
    assert compaction is not None
    assert compaction["upto_message_id"] == conversation[-7]["id"]
    assert "backend-generated compact summary" in compaction["summary"]
