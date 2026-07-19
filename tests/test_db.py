import sqlite3

import pytest

from backhead import db


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
