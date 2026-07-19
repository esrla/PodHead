import sqlite3

import pytest

from backhead import db


def _conn():
    conn = sqlite3.connect(":memory:")
    db.init_db(conn)
    return conn


def test_insert_message_with_content_and_get_conversation():
    conn = _conn()
    thread = db.create_email_thread(conn, "alice@example.com")
    mid = db.insert_message_with_content(
        conn,
        channel="email",
        thread_id=str(thread),
        sender_id="alice@example.com",
        role="user",
        timestamp="2026-01-01T00:00:00+00:00",
        content_parts=[("text", "hello"), ("image", "media/x.png")],
    )
    convo = db.get_conversation(conn, "email", str(thread))
    assert len(convo) == 1
    msg = convo[0]
    assert msg["id"] == mid
    assert msg["role"] == "user"
    assert msg["sender_id"] == "alice@example.com"
    assert [c["content_type"] for c in msg["content"]] == ["text", "image"]
    assert [c["ordinal"] for c in msg["content"]] == [0, 1]
    assert msg["content"][1]["content"] == "media/x.png"


def test_insert_message_with_content_is_atomic():
    conn = _conn()
    thread = db.create_email_thread(conn, "alice@example.com")
    # Duplicate ordinal via UNIQUE(message_id, ordinal) is impossible through the
    # helper, so force a failure by injecting a bad content tuple.
    with pytest.raises(Exception):
        db.insert_message_with_content(
            conn,
            channel="email",
            thread_id=str(thread),
            sender_id="alice@example.com",
            role="user",
            timestamp="2026-01-01T00:00:00+00:00",
            content_parts=[("text", "ok"), ("text", None)],  # None violates NOT NULL
        )
    # Nothing should be stored.
    assert db.get_conversation(conn, "email", str(thread)) == []


def test_get_conversation_orders_by_id_not_timestamp():
    conn = _conn()
    thread = db.create_email_thread(conn, "alice@example.com")
    a = db.insert_message_with_content(
        conn, channel="email", thread_id=str(thread), sender_id="a",
        role="user", timestamp="2026-01-01T00:00:99+00:00",
        content_parts=[("text", "later ts, inserted first")],
    )
    b = db.insert_message_with_content(
        conn, channel="email", thread_id=str(thread), sender_id="a",
        role="assistant", timestamp="2026-01-01T00:00:00+00:00",
        content_parts=[("text", "earlier ts, inserted second")],
    )
    ids = [m["id"] for m in db.get_conversation(conn, "email", str(thread))]
    assert ids == [a, b]


def test_email_thread_lookup_by_message_id():
    conn = _conn()
    thread = db.create_email_thread(conn, "alice@example.com")
    mid = db.insert_message_with_content(
        conn, channel="email", thread_id=str(thread), sender_id="alice@example.com",
        role="user", timestamp=db.now_local_iso(), content_parts=[("text", "hi")],
    )
    db.insert_email_message_meta(
        conn, message_id=mid, email_message_id="<m1@example.com>", process_state=db.PENDING
    )
    assert db.get_email_thread_by_message_id(conn, "<m1@example.com>") == thread
    assert db.get_email_thread_by_message_id(conn, "<nope@example.com>") is None


def test_resolve_email_thread_from_references_matches_sender():
    conn = _conn()
    thread = db.create_email_thread(conn, "alice@example.com")
    mid = db.insert_message_with_content(
        conn, channel="email", thread_id=str(thread), sender_id="alice@example.com",
        role="user", timestamp=db.now_local_iso(), content_parts=[("text", "hi")],
    )
    db.insert_email_message_meta(
        conn, message_id=mid, email_message_id="<root@example.com>", process_state=db.COMPLETED
    )
    found = db.resolve_email_thread_from_references(
        conn, sender_id="alice@example.com", in_reply_to="<root@example.com>", references=[]
    )
    assert found == thread
    other = db.resolve_email_thread_from_references(
        conn, sender_id="bob@example.com", in_reply_to="<root@example.com>", references=[]
    )
    assert other is None


def test_has_incoming_email_message_id():
    conn = _conn()
    thread = db.create_email_thread(conn, "alice@example.com")
    mid = db.insert_message_with_content(
        conn, channel="email", thread_id=str(thread), sender_id="alice@example.com",
        role="user", timestamp=db.now_local_iso(), content_parts=[("text", "hi")],
    )
    db.insert_email_message_meta(
        conn, message_id=mid, email_message_id="<dup@example.com>", process_state=db.PENDING
    )
    assert db.has_incoming_email_message_id(conn, "<dup@example.com>") is True
    assert db.has_incoming_email_message_id(conn, "<other@example.com>") is False


def _queue(conn, thread, msg_id, text, state=db.PENDING):
    mid = db.insert_message_with_content(
        conn, channel="email", thread_id=str(thread), sender_id="alice@example.com",
        role="user", timestamp=db.now_local_iso(), content_parts=[("text", text)],
    )
    db.insert_email_message_meta(
        conn, message_id=mid, email_message_id=msg_id, subject="s", process_state=state
    )
    return mid


def test_list_email_threads_with_work_and_claim():
    conn = _conn()
    thread = db.create_email_thread(conn, "alice@example.com")
    first = _queue(conn, thread, "<a@example.com>", "first")
    _queue(conn, thread, "<b@example.com>", "second")
    assert db.list_email_threads_with_work(conn) == [thread]

    claimed = db.claim_next_pending_email_message(conn, thread)
    assert claimed["id"] == first
    assert claimed["email_message_id"] == "<a@example.com>"
    assert claimed["subject"] == "s"
    assert claimed["content"][0]["content"] == "first"
    # State moved to processing
    assert db.get_email_message_meta(conn, first)["process_state"] == db.PROCESSING


def test_reset_processing_and_requeue_failed():
    conn = _conn()
    thread = db.create_email_thread(conn, "alice@example.com")
    mid = _queue(conn, thread, "<a@example.com>", "work", state=db.PROCESSING)
    db.reset_processing_email_messages(conn)
    assert db.get_email_message_meta(conn, mid)["process_state"] == db.PENDING

    db.update_email_message_state(conn, mid, db.FAILED, "boom")
    db.requeue_failed_email_messages(conn)
    assert db.get_email_message_meta(conn, mid)["process_state"] == db.PENDING


def test_convert_to_local_iso_none_returns_now():
    assert db.convert_to_local_iso(None)
    iso = db.convert_to_local_iso(0)
    assert "T" in iso
