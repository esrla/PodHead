import sqlite3
import datetime
from email.message import EmailMessage

import pytest

from backhead import db
from backhead.mail import (
    ContentPart,
    IncomingEmail,
    IMAPPollError,
    _extract_content_parts,
    _html_to_text,
    build_reply_email,
    decode_thread_id_from_message_id,
    deterministic_thread_id,
    normalize_message_id,
    parse_mime_message,
    poll_inbox,
    resolve_thread_id,
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


def test_new_thread_id_is_deterministic_from_sender_and_message_id():
    conn = _conn()
    result = store_incoming_email(
        conn=conn,
        incoming=_text_email(
            "Alice Example <ALICE@example.com>",
            "Hello",
            "First",
            message_id="<m1@example.com>",
        ),
        whitelist={"alice@example.com"},
    )
    expected = deterministic_thread_id("alice@example.com", "<m1@example.com>")
    assert result["status"] == "queued"
    assert result["thread_id"] == expected


def test_reply_headers_include_thread_id_marker_and_decode():
    thread_id = "a" * 64
    outgoing, outgoing_message_id = build_reply_email(
        from_address="podhead@example.com",
        to="alice@example.com",
        subject="Hello",
        body="Reply",
        thread_id=thread_id,
        incoming_message_id="<incoming@example.com>",
        references_header="<r1@example.com>",
    )
    assert outgoing.headers["In-Reply-To"] == "<incoming@example.com>"
    assert outgoing.headers["References"].endswith("<incoming@example.com>")
    assert decode_thread_id_from_message_id(outgoing_message_id) == thread_id


def test_thread_resolution_prefers_marker_then_new_deterministic_id():
    sender = "alice@example.com"
    seed = deterministic_thread_id(sender, "<root@example.com>")
    marked = normalize_message_id(f"<podhead.{seed}.token@example.com>")
    assert resolve_thread_id(
        sender_id=sender,
        incoming_message_id="<new@example.com>",
        in_reply_to=marked,
        references=[],
    ) == seed

    assert resolve_thread_id(
        sender_id=sender,
        incoming_message_id="<new@example.com>",
        in_reply_to=None,
        references=[],
    ) == deterministic_thread_id(sender, "<new@example.com>")


def test_non_whitelisted_sender_is_ignored_before_storage():
    conn = _conn()
    result = store_incoming_email(
        conn=conn,
        incoming=_text_email("eve@example.com", "Blocked", "Blocked", message_id="<blocked@example.com>"),
        whitelist={"alice@example.com"},
    )
    assert result["status"] == "ignored_non_whitelisted_sender"
    assert db.get_conversation(conn, "email", "any") == []


def test_store_incoming_email_returns_transport_data():
    conn = _conn()
    result = store_incoming_email(
        conn=conn,
        incoming=_text_email("alice@example.com", "Hello", "Body", message_id="<m1@example.com>"),
        whitelist={"alice@example.com"},
        imap_identifier="42",
    )
    transport = result["transport"]
    assert transport.imap_identifier == "42"
    assert transport.sender_id == "alice@example.com"
    assert transport.thread_id == result["thread_id"]


class _FakeIMAP:
    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def login(self, user, password):
        self.calls.append(("login", user))
        return "OK", []

    def select(self, inbox):
        self.calls.append(("select", inbox))
        return "OK", []

    def uid(self, command, uid, *args):
        uid_text = uid.decode() if isinstance(uid, bytes) else uid
        self.calls.append(("uid", command, uid_text, args))
        if command == "SEARCH":
            return "OK", [b"1 2"]
        if command == "FETCH" and uid_text == "1":
            if "HEADER.FIELDS (FROM)" in args[0]:
                return "OK", [(b"meta", b"From: Eve <eve@example.com>\r\n\r\n")]
            raise AssertionError("non-whitelisted sender must not be fully fetched")
        if command == "FETCH" and uid_text == "2":
            if "HEADER.FIELDS (FROM)" in args[0]:
                return "OK", [(b"meta", b"From: Alice <alice@example.com>\r\n\r\n")]
            if args[0] == "(BODY.PEEK[])":
                msg = EmailMessage()
                msg["From"] = "Alice <alice@example.com>"
                msg["Subject"] = "Hello"
                msg["Message-ID"] = "<a@example.com>"
                msg.set_content("body")
                return "OK", [(b"meta", msg.as_bytes())]
        if command in {"MOVE", "STORE"}:
            return "OK", []
        if command == "COPY":
            return "NO", []
        return "OK", []

    def expunge(self):
        self.calls.append(("expunge",))
        return "OK", []


def test_poll_inbox_uses_uid_search_and_header_first_whitelist_check(monkeypatch):
    conn = _conn()
    fake = _FakeIMAP()
    monkeypatch.setattr("backhead.mail._open_imap_connection", lambda config: fake)

    class _IMAPConfig:
        host = "imap"
        port = 993
        username = "u"
        password = "p"
        inbox = "INBOX"
        use_ssl = True

    results = poll_inbox(
        conn=conn,
        whitelist={"alice@example.com"},
        imap_config=_IMAPConfig(),
        spam_mailbox="Junk",
    )

    # UID SEARCH must be used, not plain SEARCH
    assert ("uid", "SEARCH", None, ("UNSEEN",)) in fake.calls
    assert any(result["status"] == "moved_to_spam" for result in results)
    queued = [r for r in results if r["status"] == "queued"]
    assert len(queued) == 1
    history = db.get_conversation(conn, "email", queued[0]["thread_id"])
    assert len(history) == 1

    # Full fetch must use BODY.PEEK[] not RFC822
    fetch_calls = [c for c in fake.calls if c[0] == "uid" and c[1] == "FETCH"]
    full_fetches = [c for c in fetch_calls if "(BODY.PEEK[])" in c[3]]
    assert len(full_fetches) >= 1

    # Seen flag must only be set for the whitelisted message (uid "2")
    seen_calls = [
        c for c in fake.calls
        if c[0] == "uid" and c[1] == "STORE" and "Seen" in str(c[3])
    ]
    assert len(seen_calls) == 1
    assert seen_calls[0][2] == "2"


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


# ---------------------------------------------------------------------------
# UID consistency
# ---------------------------------------------------------------------------


def test_uid_search_returns_uids_used_by_subsequent_operations(monkeypatch):
    """UIDs from UID SEARCH must be passed to UID FETCH and UID STORE."""
    conn = _conn()
    fake = _FakeIMAP()
    monkeypatch.setattr("backhead.mail._open_imap_connection", lambda config: fake)

    class _IMAPConfig:
        host = "imap"; port = 993; username = "u"; password = "p"
        inbox = "INBOX"; use_ssl = True

    poll_inbox(conn=conn, whitelist={"alice@example.com"},
               imap_config=_IMAPConfig(), spam_mailbox="Junk")

    uid_commands = [c for c in fake.calls if c[0] == "uid"]
    commands_used = {c[1] for c in uid_commands}
    assert "SEARCH" in commands_used
    assert "FETCH" in commands_used
    assert "STORE" in commands_used


# ---------------------------------------------------------------------------
# BODY.PEEK[] – no implicit Seen
# ---------------------------------------------------------------------------


def test_header_fetch_uses_body_peek_not_rfc822(monkeypatch):
    """Header fetch must use BODY.PEEK[HEADER.FIELDS ...] to avoid setting Seen."""
    conn = _conn()
    fake = _FakeIMAP()
    monkeypatch.setattr("backhead.mail._open_imap_connection", lambda config: fake)

    class _IMAPConfig:
        host = "imap"; port = 993; username = "u"; password = "p"
        inbox = "INBOX"; use_ssl = True

    poll_inbox(conn=conn, whitelist={"alice@example.com"},
               imap_config=_IMAPConfig(), spam_mailbox="Junk")

    header_fetches = [
        c for c in fake.calls
        if c[0] == "uid" and c[1] == "FETCH" and "HEADER.FIELDS" in str(c[3])
    ]
    for call in header_fetches:
        assert "BODY.PEEK" in str(call[3])
        assert "RFC822" not in str(call[3])


def test_full_fetch_uses_body_peek_not_rfc822(monkeypatch):
    """Full MIME fetch must use BODY.PEEK[] to avoid implicitly setting Seen."""
    conn = _conn()
    fake = _FakeIMAP()
    monkeypatch.setattr("backhead.mail._open_imap_connection", lambda config: fake)

    class _IMAPConfig:
        host = "imap"; port = 993; username = "u"; password = "p"
        inbox = "INBOX"; use_ssl = True

    poll_inbox(conn=conn, whitelist={"alice@example.com"},
               imap_config=_IMAPConfig(), spam_mailbox="Junk")

    full_fetches = [
        c for c in fake.calls
        if c[0] == "uid" and c[1] == "FETCH" and "(BODY.PEEK[])" in str(c[3])
    ]
    assert len(full_fetches) >= 1
    for call in full_fetches:
        assert "RFC822" not in str(call[3])


def test_seen_set_only_after_successful_storage(monkeypatch):
    r"""\\Seen must be set after the DB row is committed, not before."""
    conn = _conn()
    fake = _FakeIMAP()
    monkeypatch.setattr("backhead.mail._open_imap_connection", lambda config: fake)

    class _IMAPConfig:
        host = "imap"; port = 993; username = "u"; password = "p"
        inbox = "INBOX"; use_ssl = True

    results = poll_inbox(conn=conn, whitelist={"alice@example.com"},
                         imap_config=_IMAPConfig(), spam_mailbox="Junk")

    queued = [r for r in results if r["status"] == "queued"]
    assert len(queued) == 1
    seen_idx = next(
        i for i, c in enumerate(fake.calls)
        if c[0] == "uid" and c[1] == "STORE" and "Seen" in str(c[3])
    )
    history = db.get_conversation(conn, "email", queued[0]["thread_id"])
    assert len(history) == 1
    fetch_indices = [
        i for i, c in enumerate(fake.calls)
        if c[0] == "uid" and c[1] == "FETCH" and c[2] == "2"
    ]
    assert all(fi < seen_idx for fi in fetch_indices)


def test_storage_failure_leaves_message_unseen(monkeypatch):
    r"""If the DB insert fails, \\Seen must never be set on the IMAP message."""
    conn = sqlite3.connect(":memory:")
    # No db.init_db(conn) -> insert will fail

    fake = _FakeIMAP()
    monkeypatch.setattr("backhead.mail._open_imap_connection", lambda config: fake)

    class _IMAPConfig:
        host = "imap"; port = 993; username = "u"; password = "p"
        inbox = "INBOX"; use_ssl = True

    with pytest.raises(IMAPPollError):
        poll_inbox(conn=conn, whitelist={"alice@example.com"},
                   imap_config=_IMAPConfig(), spam_mailbox="Junk")

    seen_calls = [
        c for c in fake.calls
        if c[0] == "uid" and c[1] == "STORE" and "Seen" in str(c[3])
    ]
    assert len(seen_calls) == 0


# ---------------------------------------------------------------------------
# Unique fallback thread ID
# ---------------------------------------------------------------------------


def test_unrelated_messages_without_message_id_get_different_thread_ids():
    """Two messages from the same sender with no Message-ID must get distinct thread IDs."""
    conn = _conn()
    email1 = IncomingEmail(
        from_header="alice@example.com",
        subject="First",
        content_parts=[ContentPart(kind="text", text="msg 1")],
        message_id=None,
    )
    email2 = IncomingEmail(
        from_header="alice@example.com",
        subject="Second",
        content_parts=[ContentPart(kind="text", text="msg 2")],
        message_id=None,
    )
    r1 = store_incoming_email(conn=conn, incoming=email1, whitelist={"alice@example.com"})
    r2 = store_incoming_email(conn=conn, incoming=email2, whitelist={"alice@example.com"})
    assert r1["status"] == "queued"
    assert r2["status"] == "queued"
    assert r1["thread_id"] != r2["thread_id"]


# ---------------------------------------------------------------------------
# Audio and video placeholders
# ---------------------------------------------------------------------------


def test_audio_placeholder_exact_text():
    msg = EmailMessage()
    msg.set_content("text part")
    msg.add_attachment(b"\x00" * 16, maintype="audio", subtype="mpeg", filename="a.mp3")
    parts = _extract_content_parts(msg)
    texts = [p.text for p in parts if p.kind == "text"]
    assert "Audio detected. STT not yet implemented" in texts


def test_video_placeholder_exact_text():
    msg = EmailMessage()
    msg.set_content("text part")
    msg.add_attachment(b"\x00" * 16, maintype="video", subtype="mp4", filename="v.mp4")
    parts = _extract_content_parts(msg)
    texts = [p.text for p in parts if p.kind == "text"]
    assert "Video detected. Video normalization not yet implemented" in texts


def test_audio_video_placeholders_at_correct_position():
    """Audio/video placeholders appear as text parts in natural MIME order."""
    msg = EmailMessage()
    msg.set_content("intro")
    msg.add_attachment(b"\x00" * 16, maintype="audio", subtype="mpeg", filename="a.mp3")
    msg.add_attachment(b"\x00" * 16, maintype="video", subtype="mp4", filename="v.mp4")
    parts = _extract_content_parts(msg)
    texts = [p.text for p in parts if p.kind == "text"]
    assert "Audio detected. STT not yet implemented" in texts
    assert "Video detected. Video normalization not yet implemented" in texts
    intro_idx = texts.index("intro")
    audio_idx = texts.index("Audio detected. STT not yet implemented")
    video_idx = texts.index("Video detected. Video normalization not yet implemented")
    assert intro_idx < audio_idx < video_idx


def test_audio_video_stored_as_text_parts():
    """Audio and video placeholders must be stored as text content rows."""
    conn = _conn()
    msg = EmailMessage()
    msg["From"] = "alice@example.com"
    msg["Subject"] = "media"
    msg["Message-ID"] = "<media@example.com>"
    msg.set_content("hello")
    msg.add_attachment(b"\x00" * 16, maintype="audio", subtype="mpeg", filename="a.mp3")
    msg.add_attachment(b"\x00" * 16, maintype="video", subtype="mp4", filename="v.mp4")
    incoming = parse_mime_message(msg.as_bytes())
    result = store_incoming_email(conn=conn, incoming=incoming,
                                  whitelist={"alice@example.com"})
    assert result["status"] == "queued"
    convo = db.get_conversation(conn, "email", result["thread_id"])
    content_texts = [c["content"] for c in convo[0]["content"] if c["content_type"] == "text"]
    assert "Audio detected. STT not yet implemented" in content_texts
    assert "Video detected. Video normalization not yet implemented" in content_texts


# ---------------------------------------------------------------------------
# Date header parsing
# ---------------------------------------------------------------------------


def test_parse_mime_date_header_with_timezone():
    msg = EmailMessage()
    msg["From"] = "alice@example.com"
    msg["Date"] = "Mon, 01 Jan 2024 12:00:00 +0200"
    msg.set_content("body")
    result = parse_mime_message(msg.as_bytes())
    assert result.timestamp is not None
    assert isinstance(result.timestamp, float)
    expected = datetime.datetime(2024, 1, 1, 10, 0, 0,
                                 tzinfo=datetime.timezone.utc).timestamp()
    assert abs(result.timestamp - expected) < 2.0


def test_parse_mime_date_header_missing_uses_none():
    msg = EmailMessage()
    msg["From"] = "alice@example.com"
    msg.set_content("body")
    result = parse_mime_message(msg.as_bytes())
    assert result.timestamp is None


def test_parse_mime_date_header_invalid_uses_none():
    msg = EmailMessage()
    msg["From"] = "alice@example.com"
    msg["Date"] = "not a valid date string"
    msg.set_content("body")
    result = parse_mime_message(msg.as_bytes())
    assert result.timestamp is None


def test_store_email_with_date_uses_parsed_timestamp():
    """A valid Date header is reflected in the stored ISO timestamp."""
    conn = _conn()
    msg = EmailMessage()
    msg["From"] = "alice@example.com"
    msg["Subject"] = "dated"
    msg["Message-ID"] = "<dated@example.com>"
    msg["Date"] = "Mon, 01 Jan 2024 12:00:00 +0000"
    msg.set_content("hello")
    incoming = parse_mime_message(msg.as_bytes())
    assert incoming.timestamp is not None
    result = store_incoming_email(conn=conn, incoming=incoming,
                                  whitelist={"alice@example.com"})
    assert result["status"] == "queued"
    convo = db.get_conversation(conn, "email", result["thread_id"])
    assert convo[0]["timestamp"].startswith("2024")


def test_missing_date_falls_back_to_current_time():
    """When Date is absent the stored timestamp must be close to now."""
    conn = _conn()
    before = datetime.datetime.now().astimezone().replace(microsecond=0)
    result = store_incoming_email(
        conn=conn,
        incoming=_text_email("alice@example.com", "no date", "body",
                              message_id="<nodate@example.com>"),
        whitelist={"alice@example.com"},
    )
    after = datetime.datetime.now().astimezone() + datetime.timedelta(seconds=1)
    assert result["status"] == "queued"
    convo = db.get_conversation(conn, "email", result["thread_id"])
    stored = datetime.datetime.fromisoformat(convo[0]["timestamp"])
    assert before <= stored <= after
