import sqlite3
from email.message import EmailMessage

from backhead import db
from backhead.mail import (
    ContentPart,
    IncomingEmail,
    _extract_content_parts,
    _html_to_text,
    build_reply_email,
    process_incoming_email,
    store_incoming_email,
)


def _conn():
    conn = sqlite3.connect(":memory:")
    db.init_db(conn)
    return conn


def _text_email(from_header, subject, text, **kwargs):
    return IncomingEmail(
        from_header=from_header,
        subject=subject,
        content_parts=[ContentPart(kind="text", text=text)],
        **kwargs,
    )


def test_allowed_sender_is_queued_and_duplicate_is_ignored():
    conn = _conn()
    first = store_incoming_email(
        conn=conn,
        incoming=_text_email(
            "Alice Example <ALICE@example.com>", "Hello", "First",
            message_id="<m1@example.com>",
        ),
        whitelist={"alice@example.com"},
    )
    duplicate = store_incoming_email(
        conn=conn,
        incoming=_text_email(
            "alice@example.com", "Hello again", "Duplicate",
            message_id="<m1@example.com>",
        ),
        whitelist={"alice@example.com"},
    )

    assert first["status"] == "queued"
    assert duplicate["status"] == "ignored_duplicate_message"
    meta = db.get_email_message_meta(conn, first["message_id"])
    assert meta["process_state"] == db.PENDING


def test_in_reply_to_reuses_existing_thread_for_same_sender():
    conn = _conn()
    first = store_incoming_email(
        conn=conn,
        incoming=_text_email(
            "alice@example.com", "Hello", "First", message_id="<m1@example.com>"
        ),
        whitelist={"alice@example.com"},
    )
    second = store_incoming_email(
        conn=conn,
        incoming=_text_email(
            "alice@example.com", "Re: Hello", "Second",
            message_id="<m2@example.com>", in_reply_to="<m1@example.com>",
        ),
        whitelist={"alice@example.com"},
    )
    assert first["email_thread_id"] == second["email_thread_id"]


def test_different_sender_cannot_join_thread_via_references():
    conn = _conn()
    first = store_incoming_email(
        conn=conn,
        incoming=_text_email(
            "alice@example.com", "Hello", "A1", message_id="<shared@example.com>"
        ),
        whitelist={"alice@example.com", "bob@example.com"},
    )
    second = store_incoming_email(
        conn=conn,
        incoming=_text_email(
            "bob@example.com", "Re: Hello", "B1",
            message_id="<b1@example.com>",
            in_reply_to="<shared@example.com>",
            references="<shared@example.com>",
        ),
        whitelist={"alice@example.com", "bob@example.com"},
    )
    assert first["email_thread_id"] != second["email_thread_id"]


def test_reply_headers_include_message_id_in_reply_to_and_references():
    outgoing, outgoing_message_id = build_reply_email(
        from_address="podhead@example.com",
        incoming=_text_email(
            "alice@example.com", "Hello", "First",
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

    def raising_agent(history, current_message):
        raise RuntimeError("boom")

    try:
        process_incoming_email(
            conn=conn,
            incoming=_text_email(
                "alice@example.com", "Hello", "First", message_id="<m1@example.com>"
            ),
            whitelist={"alice@example.com"},
            run_agent=raising_agent,
            send_reply=lambda outgoing: None,
        )
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("Expected RuntimeError")

    history = db.get_conversation(conn, "email", "1")
    meta = db.get_email_message_meta(conn, history[0]["id"])
    assert meta["process_state"] == db.FAILED
    assert meta["failure_details"] == "boom"


def test_process_incoming_email_stores_reply_and_completes():
    conn = _conn()

    def agent(history, current_message):
        return "hi back"

    sent = []
    result = process_incoming_email(
        conn=conn,
        incoming=_text_email(
            "alice@example.com", "Hello", "First", message_id="<m1@example.com>"
        ),
        whitelist={"alice@example.com"},
        run_agent=agent,
        send_reply=lambda outgoing: sent.append(outgoing),
        generated_message_id="<out@example.com>",
    )
    assert result["status"] == "processed"
    assert result["outgoing_message_id"] == "<out@example.com>"
    history = db.get_conversation(conn, "email", "1")
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[1]["content"][0]["content"] == "hi back"
    assert len(sent) == 1


def test_deterministic_history_ordering_uses_id():
    conn = _conn()
    thread_id = db.create_email_thread(conn, "alice@example.com", created_ts="2026-01-01T00:00:00+00:00")
    first_id = db.insert_message_with_content(
        conn, channel="email", thread_id=str(thread_id), sender_id="alice@example.com",
        role="user", timestamp="2026-01-01T00:00:10+00:00", content_parts=[("text", "first")],
    )
    second_id = db.insert_message_with_content(
        conn, channel="email", thread_id=str(thread_id), sender_id="alice@example.com",
        role="user", timestamp="2026-01-01T00:00:10+00:00", content_parts=[("text", "second")],
    )
    ordered_ids = [row["id"] for row in db.get_conversation(conn, "email", str(thread_id))]
    assert ordered_ids == [first_id, second_id]


def test_email_message_meta_includes_subject_and_state():
    conn = _conn()
    thread_id = db.create_email_thread(conn, "alice@example.com")
    message_id = db.insert_message_with_content(
        conn, channel="email", thread_id=str(thread_id), sender_id="alice@example.com",
        role="user", timestamp=db.now_local_iso(), content_parts=[("text", "body")],
    )
    db.insert_email_message_meta(
        conn, message_id=message_id, email_message_id="<m@example.com>",
        subject="subject", process_state=db.FAILED, failure_details="details",
    )
    meta = db.get_email_message_meta(conn, message_id)
    assert meta["subject"] == "subject"
    assert meta["process_state"] == db.FAILED
    assert meta["failure_details"] == "details"


def test_parse_mime_prefers_plain_text_and_skips_attachments():
    message = EmailMessage()
    message.set_content("plain text")
    message.add_alternative("<p>html text</p>", subtype="html")
    message.add_attachment(b"bytes", maintype="application", subtype="octet-stream", filename="file.bin")
    parts = _extract_content_parts(message)
    texts = [p.text for p in parts if p.kind == "text"]
    assert "plain text" in texts
    assert not any("html text" in t for t in texts)


def test_parse_mime_uses_html_when_plain_text_missing():
    message = EmailMessage()
    message.set_content("<p>html only</p>", subtype="html")
    parts = _extract_content_parts(message)
    texts = [p.text for p in parts if p.kind == "text"]
    assert texts == ["html only"]


def test_html_to_text_strips_tags_and_scripts():
    html = "<html><head><style>x{}</style></head><body><p>Hello</p><script>bad()</script><p>World</p></body></html>"
    assert _html_to_text(html) == "Hello\nWorld"


def test_parse_mime_extracts_inline_image_as_bytes():
    message = EmailMessage()
    message.make_mixed()
    text = EmailMessage()
    text.set_content("look at this")
    message.attach(text)
    img = EmailMessage()
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 20
    img.set_content(png, maintype="image", subtype="png")
    message.attach(img)
    parts = _extract_content_parts(message)
    kinds = [p.kind for p in parts]
    assert "image_bytes" in kinds
    image_part = next(p for p in parts if p.kind == "image_bytes")
    assert image_part.image_bytes == png


def test_cid_image_not_duplicated():
    message = EmailMessage()
    message.make_mixed()
    text = EmailMessage()
    text.set_content("body")
    message.attach(text)
    png = b"\x89PNG\r\n\x1a\n" + b"1" * 20
    img = EmailMessage()
    img.set_content(png, maintype="image", subtype="png")
    img["Content-ID"] = "<abc123>"
    message.attach(img)
    img2 = EmailMessage()
    img2.set_content(png, maintype="image", subtype="png")
    img2["Content-ID"] = "<abc123>"
    message.attach(img2)
    parts = _extract_content_parts(message)
    image_parts = [p for p in parts if p.kind == "image_bytes"]
    assert len(image_parts) == 1


def test_store_image_without_media_root_uses_placeholder():
    conn = _conn()
    png = b"\x89PNG\r\n\x1a\n" + b"2" * 20
    result = store_incoming_email(
        conn=conn,
        incoming=IncomingEmail(
            from_header="alice@example.com",
            subject="pic",
            content_parts=[
                ContentPart(kind="text", text="see"),
                ContentPart(kind="image_bytes", image_bytes=png, mime_type="image/png"),
            ],
            message_id="<img@example.com>",
        ),
        whitelist={"alice@example.com"},
    )
    history = db.get_conversation(conn, "email", str(result["email_thread_id"]))
    content = history[0]["content"]
    assert content[0]["content_type"] == "text"
    assert content[1]["content_type"] == "text"
    assert content[1]["content"] == "[image omitted]"


def test_store_image_with_media_root_saves_file(tmp_path):
    conn = _conn()
    png = b"\x89PNG\r\n\x1a\n" + b"3" * 20
    result = store_incoming_email(
        conn=conn,
        incoming=IncomingEmail(
            from_header="alice@example.com",
            subject="pic",
            content_parts=[
                ContentPart(kind="image_bytes", image_bytes=png, mime_type="image/png"),
            ],
            message_id="<img2@example.com>",
        ),
        whitelist={"alice@example.com"},
        media_root=tmp_path,
    )
    history = db.get_conversation(conn, "email", str(result["email_thread_id"]))
    content = history[0]["content"]
    assert content[0]["content_type"] == "image"
    rel = content[0]["content"]
    assert rel.startswith("media/")
    assert (tmp_path / rel).exists()


def test_non_whitelisted_sender_is_ignored_before_storage():
    conn = _conn()
    result = store_incoming_email(
        conn=conn,
        incoming=_text_email(
            "eve@example.com", "Blocked", "Blocked", message_id="<blocked@example.com>"
        ),
        whitelist={"alice@example.com"},
    )
    assert result["status"] == "ignored_non_whitelisted_sender"
    assert db.list_email_threads_with_work(conn) == []


def test_incoming_email_body_property_concatenates_text():
    incoming = IncomingEmail(
        from_header="a@b.com",
        subject="s",
        content_parts=[
            ContentPart(kind="text", text="one"),
            ContentPart(kind="image_bytes", image_bytes=b"x"),
            ContentPart(kind="text", text="two"),
        ],
    )
    assert incoming.body == "one\n\ntwo"
