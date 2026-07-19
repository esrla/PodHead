import sqlite3
from email.message import EmailMessage

from backhead import db
from backhead.mail import (
    ContentPart,
    IncomingEmail,
    _extract_content_parts,
    _html_to_text,
    build_reply_email,
    decode_thread_id_from_message_id,
    deterministic_thread_id,
    normalize_message_id,
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

    def search(self, charset, criterion):
        self.calls.append(("search", criterion))
        return "OK", [b"1 2"]

    def uid(self, command, uid, *args):
        uid_text = uid.decode() if isinstance(uid, bytes) else uid
        self.calls.append(("uid", command, uid_text, args))
        if command == "FETCH" and uid_text == "1":
            if "HEADER.FIELDS (FROM)" in args[0]:
                return "OK", [(b"meta", b"From: Eve <eve@example.com>\r\n\r\n")]
            raise AssertionError("non-whitelisted sender must not fetch RFC822")
        if command == "FETCH" and uid_text == "2":
            if "HEADER.FIELDS (FROM)" in args[0]:
                return "OK", [(b"meta", b"From: Alice <alice@example.com>\r\n\r\n")]
            if args[0] == "(RFC822)":
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


def test_poll_inbox_uses_unseen_and_header_first_whitelist_check(monkeypatch):
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

    assert ("search", "UNSEEN") in fake.calls
    assert any(result["status"] == "moved_to_spam" for result in results)
    queued = [r for r in results if r["status"] == "queued"]
    assert len(queued) == 1
    history = db.get_conversation(conn, "email", queued[0]["thread_id"])
    assert len(history) == 1


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
