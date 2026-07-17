import sqlite3

from backhead import db
from backhead.mail import IncomingEmail, process_incoming_email


def _conn():
    conn = sqlite3.connect(":memory:")
    db.init_db(conn)
    return conn


def _count(conn, table):
    if table not in {"conversations", "messages"}:
        raise ValueError(f"Unsupported table: {table}")
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_allowed_sender_standalone_email_creates_new_conversation():
    conn = _conn()
    sent = []

    result = process_incoming_email(
        conn=conn,
        incoming=IncomingEmail(
            from_header="Alice Example <ALICE@Example.com>",
            subject="Hello",
            body="First",
            message_id="<m1@example.com>",
        ),
        whitelist={"alice@example.com"},
        run_agent=lambda history, incoming: "Reply 1",
        send_reply=sent.append,
    )

    assert result["status"] == "processed"
    assert _count(conn, "conversations") == 1
    assert _count(conn, "messages") == 2
    assert len(sent) == 1


def test_in_reply_to_reuses_existing_conversation_for_same_sender():
    conn = _conn()

    first = process_incoming_email(
        conn=conn,
        incoming=IncomingEmail(
            from_header="alice@example.com",
            subject="Hello",
            body="First",
            message_id="<m1@example.com>",
        ),
        whitelist={"alice@example.com"},
        run_agent=lambda history, incoming: "Reply 1",
        send_reply=lambda outgoing: None,
    )
    second = process_incoming_email(
        conn=conn,
        incoming=IncomingEmail(
            from_header="alice@example.com",
            subject="Re: Hello",
            body="Second",
            message_id="<m2@example.com>",
            in_reply_to="<m1@example.com>",
        ),
        whitelist={"alice@example.com"},
        run_agent=lambda history, incoming: "Reply 2",
        send_reply=lambda outgoing: None,
    )

    assert first["conversation_id"] == second["conversation_id"]
    assert _count(conn, "conversations") == 1


def test_references_only_reuses_existing_conversation_for_same_sender():
    conn = _conn()

    first = process_incoming_email(
        conn=conn,
        incoming=IncomingEmail(
            from_header="alice@example.com",
            subject="Hello",
            body="First",
            message_id="<m1@example.com>",
        ),
        whitelist={"alice@example.com"},
        run_agent=lambda history, incoming: "Reply 1",
        send_reply=lambda outgoing: None,
    )
    second = process_incoming_email(
        conn=conn,
        incoming=IncomingEmail(
            from_header="alice@example.com",
            subject="Re: Hello",
            body="Second",
            message_id="<m2@example.com>",
            references="<old@example.com> <m1@example.com>",
        ),
        whitelist={"alice@example.com"},
        run_agent=lambda history, incoming: "Reply 2",
        send_reply=lambda outgoing: None,
    )

    assert first["conversation_id"] == second["conversation_id"]
    assert _count(conn, "conversations") == 1


def test_same_sender_new_standalone_email_creates_separate_conversation():
    conn = _conn()

    first = process_incoming_email(
        conn=conn,
        incoming=IncomingEmail(
            from_header="alice@example.com",
            subject="Topic A",
            body="A1",
            message_id="<a1@example.com>",
        ),
        whitelist={"alice@example.com"},
        run_agent=lambda history, incoming: "Reply A1",
        send_reply=lambda outgoing: None,
    )
    second = process_incoming_email(
        conn=conn,
        incoming=IncomingEmail(
            from_header="alice@example.com",
            subject="Topic B",
            body="B1",
            message_id="<b1@example.com>",
        ),
        whitelist={"alice@example.com"},
        run_agent=lambda history, incoming: "Reply B1",
        send_reply=lambda outgoing: None,
    )

    assert first["conversation_id"] != second["conversation_id"]
    assert _count(conn, "conversations") == 2


def test_different_sender_cannot_join_conversation_via_references():
    conn = _conn()

    first = process_incoming_email(
        conn=conn,
        incoming=IncomingEmail(
            from_header="alice@example.com",
            subject="Hello",
            body="A1",
            message_id="<shared@example.com>",
        ),
        whitelist={"alice@example.com", "bob@example.com"},
        run_agent=lambda history, incoming: "Reply A1",
        send_reply=lambda outgoing: None,
    )
    second = process_incoming_email(
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
        run_agent=lambda history, incoming: "Reply B1",
        send_reply=lambda outgoing: None,
    )

    assert first["conversation_id"] != second["conversation_id"]
    assert _count(conn, "conversations") == 2


def test_non_whitelisted_sender_is_ignored_before_conversation_and_agent_processing():
    conn = _conn()
    calls = {"agent": 0, "reply": 0}

    def blocked_run_agent(history, incoming):
        calls["agent"] += 1
        return "should not run"

    def blocked_send_reply(outgoing):
        calls["reply"] += 1

    result = process_incoming_email(
        conn=conn,
        incoming=IncomingEmail(
            from_header="Eve <eve@example.com>",
            subject="Hello",
            body="Blocked",
            message_id="<e1@example.com>",
        ),
        whitelist={"alice@example.com"},
        run_agent=blocked_run_agent,
        send_reply=blocked_send_reply,
    )

    assert result["status"] == "ignored_non_whitelisted_sender"
    assert _count(conn, "conversations") == 0
    assert _count(conn, "messages") == 0
    assert calls == {"agent": 0, "reply": 0}


def test_duplicate_incoming_message_id_not_processed_twice():
    conn = _conn()
    calls = {"agent": 0}

    def run_agent_once(history, incoming):
        calls["agent"] += 1
        return "Reply"

    process_incoming_email(
        conn=conn,
        incoming=IncomingEmail(
            from_header="alice@example.com",
            subject="Hello",
            body="First",
            message_id="<dup@example.com>",
        ),
        whitelist={"alice@example.com"},
        run_agent=run_agent_once,
        send_reply=lambda outgoing: None,
    )
    duplicate = process_incoming_email(
        conn=conn,
        incoming=IncomingEmail(
            from_header="alice@example.com",
            subject="Hello again",
            body="Duplicate payload",
            message_id="<dup@example.com>",
        ),
        whitelist={"alice@example.com"},
        run_agent=run_agent_once,
        send_reply=lambda outgoing: None,
    )

    assert duplicate["status"] == "ignored_duplicate_message"
    assert _count(conn, "messages") == 2
    assert calls["agent"] == 1


def test_outgoing_reply_headers_include_message_id_in_reply_to_and_references():
    conn = _conn()
    sent = []

    result = process_incoming_email(
        conn=conn,
        incoming=IncomingEmail(
            from_header="alice@example.com",
            subject="Hello",
            body="First",
            message_id="<incoming@example.com>",
            references="<r1@example.com> <r2@example.com>",
        ),
        whitelist={"alice@example.com"},
        run_agent=lambda history, incoming: "Reply",
        send_reply=sent.append,
        generated_message_id="<outgoing@example.com>",
    )

    assert result["status"] == "processed"
    assert sent
    headers = sent[0].headers
    assert headers["Message-ID"] == "<outgoing@example.com>"
    assert headers["In-Reply-To"] == "<incoming@example.com>"
    assert headers["References"] == "<r1@example.com> <r2@example.com> <incoming@example.com>"

    stored = conn.execute(
        """
        SELECT email_message_id FROM messages
        WHERE direction='outgoing' AND conversation_id=?
        """,
        (result["conversation_id"],),
    ).fetchone()
    assert stored[0] == "<outgoing@example.com>"


def test_missing_or_malformed_thread_headers_create_new_conversation_without_crashing():
    conn = _conn()

    first = process_incoming_email(
        conn=conn,
        incoming=IncomingEmail(
            from_header="alice@example.com",
            subject="Hello",
            body="First",
            message_id="not-a-message-id",
            in_reply_to="still not valid",
            references="garbage",
        ),
        whitelist={"alice@example.com"},
        run_agent=lambda history, incoming: "Reply 1",
        send_reply=lambda outgoing: None,
    )
    second = process_incoming_email(
        conn=conn,
        incoming=IncomingEmail(
            from_header="alice@example.com",
            subject="Hello 2",
            body="Second",
            message_id=None,
            in_reply_to=None,
            references=None,
        ),
        whitelist={"alice@example.com"},
        run_agent=lambda history, incoming: "Reply 2",
        send_reply=lambda outgoing: None,
    )

    assert first["status"] == "processed"
    assert second["status"] == "processed"
    assert first["conversation_id"] != second["conversation_id"]
    assert _count(conn, "conversations") == 2


def test_history_loaded_for_agent_contains_only_resolved_conversation():
    conn = _conn()
    incoming_histories = []

    def run_agent(history, incoming):
        incoming_histories.append(
            [row["content"] for row in history if row["direction"] == "incoming"]
        )
        return f"Reply to {incoming.body}"

    first = process_incoming_email(
        conn=conn,
        incoming=IncomingEmail(
            from_header="alice@example.com",
            subject="Topic A",
            body="A1",
            message_id="<a1@example.com>",
        ),
        whitelist={"alice@example.com"},
        run_agent=run_agent,
        send_reply=lambda outgoing: None,
    )
    process_incoming_email(
        conn=conn,
        incoming=IncomingEmail(
            from_header="alice@example.com",
            subject="Topic B",
            body="B1",
            message_id="<b1@example.com>",
        ),
        whitelist={"alice@example.com"},
        run_agent=run_agent,
        send_reply=lambda outgoing: None,
    )
    third = process_incoming_email(
        conn=conn,
        incoming=IncomingEmail(
            from_header="alice@example.com",
            subject="Re: Topic A",
            body="A2",
            message_id="<a2@example.com>",
            in_reply_to="<a1@example.com>",
        ),
        whitelist={"alice@example.com"},
        run_agent=run_agent,
        send_reply=lambda outgoing: None,
    )

    assert first["conversation_id"] == third["conversation_id"]
    assert incoming_histories[2] == ["A1", "A2"]


def test_run_agent_exception_bubbles_and_no_reply_is_sent():
    conn = _conn()
    sent = []

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
            send_reply=sent.append,
        )
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("Expected RuntimeError to bubble up")

    assert sent == []
