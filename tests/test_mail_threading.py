import sqlite3
from email.message import EmailMessage

from backhead import db
from backhead.mail import IncomingEmail, _extract_text_body, build_reply_email, process_incoming_email, store_incoming_email



def _conn():
    conn = sqlite3.connect(":memory:")
    db.init_db(conn)
    return conn



def test_allowed_sender_is_queued_and_duplicate_is_ignored():
    conn = _conn()
    first = store_incoming_email(
        conn=conn,
        incoming=IncomingEmail(
            from_header="Alice Example <ALICE@example.com>",
            subject="Hello",
            body="First",
            message_id="<m1@example.com>",
        ),
        whitelist={"alice@example.com"},
    )
    duplicate = store_incoming_email(
        conn=conn,
        incoming=IncomingEmail(
            from_header="alice@example.com",
            subject="Hello again",
            body="Duplicate",
            message_id="<m1@example.com>",
        ),
        whitelist={"alice@example.com"},
    )

    assert first["status"] == "queued"
    assert duplicate["status"] == "ignored_duplicate_message"
    stored = db.get_message(conn, first["message_row_id"])
    assert stored["process_state"] == db.PENDING



def test_in_reply_to_reuses_existing_conversation_for_same_sender():
    conn = _conn()
    first = store_incoming_email(
        conn=conn,
        incoming=IncomingEmail(
            from_header="alice@example.com",
            subject="Hello",
            body="First",
            message_id="<m1@example.com>",
        ),
        whitelist={"alice@example.com"},
    )
    second = store_incoming_email(
        conn=conn,
        incoming=IncomingEmail(
            from_header="alice@example.com",
            subject="Re: Hello",
            body="Second",
            message_id="<m2@example.com>",
            in_reply_to="<m1@example.com>",
        ),
        whitelist={"alice@example.com"},
    )
    assert first["conversation_id"] == second["conversation_id"]



def test_different_sender_cannot_join_conversation_via_references():
    conn = _conn()
    first = store_incoming_email(
        conn=conn,
        incoming=IncomingEmail(
            from_header="alice@example.com",
            subject="Hello",
            body="A1",
            message_id="<shared@example.com>",
        ),
        whitelist={"alice@example.com", "bob@example.com"},
    )
    second = store_incoming_email(
        conn=conn,
        incoming=IncomingEmail(
            from_header="bob@example.com",
            subject="Re: Hello",
            body="B1",
            message_id="<b1@example.com>",
            in_reply_to="<shared@example.com>",
            references="<shared@example.com>",
        ),
        whitelist={"alice@example.com", "bob@example.com"},
    )
    assert first["conversation_id"] != second["conversation_id"]



def test_reply_headers_include_message_id_in_reply_to_and_references():
    outgoing, outgoing_message_id = build_reply_email(
        from_address="podhead@example.com",
        incoming=IncomingEmail(
            from_header="alice@example.com",
            subject="Hello",
            body="First",
            message_id="<incoming@example.com>",
            references="<r1@example.com> <r2@example.com>",
        ),
        to="alice@example.com",
        body="Reply",
        generated_message_id="<outgoing@example.com>",
    )

    assert outgoing_message_id == "<outgoing@example.com>"
    assert outgoing.headers["Message-ID"] == "<outgoing@example.com>"
    assert outgoing.headers["In-Reply-To"] == "<incoming@example.com>"
    assert outgoing.headers["References"] == "<r1@example.com> <r2@example.com> <incoming@example.com>"



def test_process_incoming_email_marks_failed_message_for_retry():
    conn = _conn()

    def raising_agent(history, incoming):
        raise RuntimeError("boom")

    try:
        process_incoming_email(
            conn=conn,
            incoming=IncomingEmail(
                from_header="alice@example.com",
                subject="Hello",
                body="First",
                message_id="<m1@example.com>",
            ),
            whitelist={"alice@example.com"},
            run_agent=raising_agent,
            send_reply=lambda outgoing: None,
        )
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("Expected RuntimeError")

    history = db.get_conversation_history(conn, 1)
    assert history[0]["process_state"] == db.FAILED
    assert history[0]["failure_details"] == "boom"



def test_deterministic_history_ordering_uses_timestamp_then_id():
    conn = _conn()
    conversation_id = db.create_conversation(conn, "alice@example.com", created_ts=1)
    first_id = db.insert_message(
        conn,
        conversation_id=conversation_id,
        email_message_id="<m1@example.com>",
        direction="incoming",
        content="first",
        subject="A",
        timestamp=10,
        process_state=db.COMPLETED,
    )
    second_id = db.insert_message(
        conn,
        conversation_id=conversation_id,
        email_message_id="<m2@example.com>",
        direction="incoming",
        content="second",
        subject="B",
        timestamp=10,
        process_state=db.COMPLETED,
    )
    ordered_ids = [row["id"] for row in db.get_conversation_history(conn, conversation_id)]
    assert ordered_ids == [first_id, second_id]



def test_row_to_message_conversion_includes_subject_and_state():
    conn = _conn()
    conversation_id = db.create_conversation(conn, "alice@example.com", created_ts=1)
    message_id = db.insert_message(
        conn,
        conversation_id=conversation_id,
        email_message_id="<m@example.com>",
        direction="incoming",
        content="body",
        subject="subject",
        timestamp=2,
        process_state=db.FAILED,
        failure_details="details",
    )
    row = db.get_message(conn, message_id)
    assert row["subject"] == "subject"
    assert row["process_state"] == db.FAILED
    assert row["failure_details"] == "details"



def test_extract_text_body_prefers_plain_text_and_skips_attachments():
    message = EmailMessage()
    message.set_content("plain text")
    message.add_alternative("<p>html text</p>", subtype="html")
    message.add_attachment(b"bytes", maintype="application", subtype="octet-stream", filename="file.bin")
    assert _extract_text_body(message) == "plain text"



def test_extract_text_body_uses_html_when_plain_text_missing():
    message = EmailMessage()
    message.add_alternative("<p>html only</p>", subtype="html")
    assert _extract_text_body(message) == "<p>html only</p>"



def test_non_whitelisted_sender_is_ignored_before_storage():
    conn = _conn()
    result = store_incoming_email(
        conn=conn,
        incoming=IncomingEmail(
            from_header="eve@example.com",
            subject="Blocked",
            body="Blocked",
            message_id="<blocked@example.com>",
        ),
        whitelist={"alice@example.com"},
    )
    assert result["status"] == "ignored_non_whitelisted_sender"
    assert db.list_conversations_with_work(conn) == []
