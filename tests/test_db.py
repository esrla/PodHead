import sqlite3

import pytest

from backhead import chat_history, db


def _conn():
    conn = sqlite3.connect(":memory:")
    db.init_db(conn)
    return conn


def test_insert_message_with_content_and_get_conversation():
    conn = _conn()
    mid = db.insert_message_with_content(
        conn,
        channel="email",
        thread_id="thread-1",
        sender_id="alice@example.com",
        role="user",
        timestamp="2026-01-01T00:00:00+00:00",
        content_parts=[("text", "hello"), ("image", "media/x.png")],
    )
    convo = db.get_conversation(conn, "email", "thread-1")
    assert len(convo) == 1
    msg = convo[0]
    assert msg["id"] == mid
    assert msg["role"] == "user"
    assert msg["sender_id"] == "alice@example.com"
    assert [c["content_type"] for c in msg["content"]] == ["text", "image"]
    assert [c["ordinal"] for c in msg["content"]] == [0, 1]


def test_insert_message_with_content_is_atomic():
    conn = _conn()
    with pytest.raises(Exception):
        db.insert_message_with_content(
            conn,
            channel="email",
            thread_id="thread-1",
            sender_id="alice@example.com",
            role="user",
            timestamp="2026-01-01T00:00:00+00:00",
            content_parts=[("text", "ok"), ("text", None)],
        )
    assert db.get_conversation(conn, "email", "thread-1") == []


def test_get_conversation_orders_by_id_not_timestamp():
    conn = _conn()
    a = db.insert_message_with_content(
        conn,
        channel="email",
        thread_id="thread-1",
        sender_id="a",
        role="user",
        timestamp="2026-01-01T00:00:99+00:00",
        content_parts=[("text", "later ts, inserted first")],
    )
    b = db.insert_message_with_content(
        conn,
        channel="email",
        thread_id="thread-1",
        sender_id="a",
        role="assistant",
        timestamp="2026-01-01T00:00:00+00:00",
        content_parts=[("text", "earlier ts, inserted second")],
    )
    ids = [m["id"] for m in db.get_conversation(conn, "email", "thread-1")]
    assert ids == [a, b]


def test_convert_to_local_iso_none_returns_now():
    assert db.convert_to_local_iso(None)
    iso = db.convert_to_local_iso(0)
    assert "T" in iso


def test_find_oldest_message_needing_embedding_and_store_replacement():
    conn = _conn()
    message_id = db.insert_message_with_content(
        conn,
        channel="email",
        thread_id="thread-1",
        sender_id="alice@example.com",
        role="user",
        timestamp="2026-01-01T00:00:00+00:00",
        content_parts=[("text", "hello")],
    )

    candidate = db.find_oldest_message_needing_embedding(
        conn,
        sender_id="alice@example.com",
        embedding_model="embed-model",
        searchable_text_builder=chat_history.build_searchable_message_text,
        content_hash_builder=chat_history.build_message_content_hash,
    )
    assert candidate is not None
    assert candidate["id"] == message_id

    db.store_or_replace_message_embedding(
        conn,
        message_id=message_id,
        embedding_model="embed-model",
        content_hash=candidate["content_hash"],
        searchable_text=candidate["searchable_text"],
        embedding=chat_history.vector_to_blob([1.0, 0.0]),
        embedded_at="2026-01-01T00:00:01+00:00",
    )

    assert (
        db.find_oldest_message_needing_embedding(
            conn,
            sender_id="alice@example.com",
            embedding_model="embed-model",
            searchable_text_builder=chat_history.build_searchable_message_text,
            content_hash_builder=chat_history.build_message_content_hash,
        )
        is None
    )


def test_get_message_embeddings_for_sender_filters_thread():
    conn = _conn()
    first = db.insert_message_with_content(
        conn,
        channel="email",
        thread_id="thread-1",
        sender_id="alice@example.com",
        role="user",
        timestamp="2026-01-01T00:00:00+00:00",
        content_parts=[("text", "hello")],
    )
    second = db.insert_message_with_content(
        conn,
        channel="email",
        thread_id="thread-2",
        sender_id="alice@example.com",
        role="assistant",
        timestamp="2026-01-01T00:00:01+00:00",
        content_parts=[("text", "world")],
    )
    for message_id, text in ((first, "hello"), (second, "world")):
        message = db.get_conversation(conn, "email", f"thread-{1 if message_id == first else 2}")[0]
        searchable = chat_history.build_searchable_message_text(message)
        db.store_or_replace_message_embedding(
            conn,
            message_id=message_id,
            embedding_model="embed-model",
            content_hash=chat_history.build_message_content_hash(message, searchable, "embed-model"),
            searchable_text=searchable,
            embedding=chat_history.vector_to_blob([1.0, 0.0]),
            embedded_at="2026-01-01T00:00:01+00:00",
        )

    rows = db.get_message_embeddings_for_sender(
        conn,
        sender_id="alice@example.com",
        exclude_thread_id="thread-1",
    )
    assert [row["message_id"] for row in rows] == [second]
