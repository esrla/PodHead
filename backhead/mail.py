"""Mail parsing, routing, IMAP polling, and SMTP sending helpers.

The email adapter normalizes incoming email into the generic message model
defined in :mod:`backhead.db`. An email becomes one ``messages`` row (role
``user``) plus one ``message_content`` row per ordered content part.
"""

from __future__ import annotations

from dataclasses import dataclass
from email import message_from_bytes, policy
from email.message import EmailMessage, Message
from email.utils import formataddr, make_msgid, parseaddr
from html.parser import HTMLParser
import imaplib
from pathlib import Path
import re
import smtplib
from typing import Callable, NamedTuple

from backhead import db
from backhead import media as media_mod

MESSAGE_ID_PATTERN = re.compile(r"<[^<>]+>")


class ContentPart(NamedTuple):
    kind: str  # 'text', 'image_bytes', 'audio_placeholder', 'video_placeholder'
    text: str = ""
    image_bytes: bytes = b""
    mime_type: str = ""


@dataclass(frozen=True)
class IncomingEmail:
    from_header: str
    subject: str
    content_parts: list  # list[ContentPart] — ordered content
    message_id: str | None = None
    in_reply_to: str | None = None
    references: str | None = None
    timestamp: int | float | None = None

    @property
    def body(self) -> str:
        """Return concatenation of all text parts (convenience)."""
        texts = [p.text for p in self.content_parts if p.kind == "text"]
        return "\n\n".join(t for t in texts if t)


@dataclass(frozen=True)
class OutgoingEmail:
    to: str
    from_address: str
    subject: str
    body: str
    headers: dict[str, str]


class SMTPDeliveryError(RuntimeError):
    error_type = "smtp_delivery_error"


class IMAPPollError(RuntimeError):
    error_type = "imap_poll_error"


class SMTPConfigProtocol:
    host: str
    port: int
    username: str
    password: str
    use_tls: bool


class IMAPConfigProtocol:
    host: str
    port: int
    username: str
    password: str
    inbox: str
    use_ssl: bool


def normalize_sender(from_header: str) -> str | None:
    _, address = parseaddr(from_header or "")
    normalized = (address or "").strip().lower()
    return normalized or None


def normalize_message_id(value: str | None) -> str | None:
    normalized: str | None = None
    if value:
        raw = value.strip()
        if raw:
            match = MESSAGE_ID_PATTERN.search(raw)
            if match:
                normalized = match.group(0).strip().lower()
            elif "@" in raw and " " not in raw:
                normalized = f"<{raw.lower()}>"
    return normalized


def normalize_references(value: str | None) -> list[str]:
    if not value:
        return []
    return [m.group(0).strip().lower() for m in MESSAGE_ID_PATTERN.finditer(value)]


# --------------------------------------------------------------------------- #
# HTML -> text conversion
# --------------------------------------------------------------------------- #


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True
        elif tag in ("br", "p", "div", "tr", "h1", "h2", "h3", "h4", "h5", "h6"):
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        text = "".join(self._parts)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def _html_to_text(html: str) -> str:
    extractor = _TextExtractor()
    extractor.feed(html)
    return extractor.get_text()


# --------------------------------------------------------------------------- #
# MIME parsing into ordered content parts
# --------------------------------------------------------------------------- #


def _content_id(part: Message) -> str | None:
    cid = part.get("Content-ID")
    if not cid:
        return None
    return cid.strip().strip("<>").lower()


def parse_mime_message(raw_message: bytes) -> IncomingEmail:
    """Parse raw email bytes into an :class:`IncomingEmail` with ordered parts."""
    parsed = message_from_bytes(raw_message, policy=policy.default)
    parts = _extract_content_parts(parsed)
    return IncomingEmail(
        from_header=parsed.get("From", ""),
        subject=parsed.get("Subject", ""),
        content_parts=parts,
        message_id=parsed.get("Message-ID"),
        in_reply_to=parsed.get("In-Reply-To"),
        references=parsed.get("References"),
        timestamp=None,
    )


def _extract_content_parts(message: Message) -> list[ContentPart]:
    parts: list[ContentPart] = []
    seen_content_ids: set[str] = set()
    html_fallback: str | None = None
    have_plain_text = False

    if not message.is_multipart():
        content = message.get_content()
        ctype = message.get_content_type()
        if ctype == "text/plain" and isinstance(content, str):
            stripped = content.strip()
            if stripped:
                parts.append(ContentPart(kind="text", text=stripped))
        elif ctype == "text/html" and isinstance(content, str):
            text = _html_to_text(content)
            if text:
                parts.append(ContentPart(kind="text", text=text))
        return parts

    for part in message.walk():
        maintype = part.get_content_maintype()
        ctype = part.get_content_type()
        if maintype == "multipart":
            continue

        if ctype == "text/plain":
            content = part.get_content()
            if isinstance(content, str) and content.strip():
                parts.append(ContentPart(kind="text", text=content.strip()))
                have_plain_text = True
        elif ctype == "text/html":
            content = part.get_content()
            if isinstance(content, str):
                text = _html_to_text(content)
                if text and html_fallback is None:
                    html_fallback = text
        elif maintype == "image":
            cid = _content_id(part)
            if cid is not None and cid in seen_content_ids:
                continue
            if cid is not None:
                seen_content_ids.add(cid)
            payload = part.get_payload(decode=True)
            if payload:
                parts.append(
                    ContentPart(
                        kind="image_bytes",
                        image_bytes=payload,
                        mime_type=ctype,
                    )
                )
        elif maintype == "audio":
            parts.append(
                ContentPart(kind="audio_placeholder", text="[audio attachment omitted]")
            )
        elif maintype == "video":
            parts.append(
                ContentPart(kind="video_placeholder", text="[video attachment omitted]")
            )
        # other types (application/*, etc.) are skipped

    if not have_plain_text and html_fallback:
        # Fallback: no plain-text part was found, so use the HTML-derived text.
        # Insert at position 0 so it appears before any inline images.
        parts.insert(0, ContentPart(kind="text", text=html_fallback))

    return parts


# --------------------------------------------------------------------------- #
# Reply header helpers
# --------------------------------------------------------------------------- #


def _subject_for_reply(subject: str) -> str:
    if not subject:
        return "Re: (no subject)"
    if subject.lower().strip().startswith("re:"):
        return subject
    return f"Re: {subject}"


def _build_references_header(
    existing_refs: list[str], incoming_message_id: str | None
) -> str | None:
    refs = list(existing_refs)
    if incoming_message_id and incoming_message_id not in refs:
        refs.append(incoming_message_id)
    if not refs:
        return None
    return " ".join(refs)


def _build_outgoing_headers(
    *,
    incoming_message_id: str | None,
    references: list[str],
    generated_message_id: str | None = None,
) -> tuple[str, dict[str, str]]:
    outgoing_message_id = normalize_message_id(generated_message_id or make_msgid())
    if outgoing_message_id is None:
        raise ValueError("Failed to generate a valid Message-ID for outgoing email")
    headers: dict[str, str] = {"Message-ID": outgoing_message_id}
    if incoming_message_id:
        headers["In-Reply-To"] = incoming_message_id
    references_header = _build_references_header(references, incoming_message_id)
    if references_header:
        headers["References"] = references_header
    return outgoing_message_id, headers


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #


def _content_parts_for_storage(
    incoming: IncomingEmail,
    media_root: Path | None,
) -> list[tuple[str, str]]:
    stored: list[tuple[str, str]] = []
    for part in incoming.content_parts:
        if part.kind == "text":
            if part.text:
                stored.append(("text", part.text))
        elif part.kind in ("audio_placeholder", "video_placeholder"):
            if part.text:
                stored.append(("text", part.text))
        elif part.kind == "image_bytes":
            if media_root is not None:
                rel = media_mod.save_image(part.image_bytes, media_root)
                if rel is not None:
                    stored.append(("image", rel))
                else:
                    stored.append(("text", "[unsupported image omitted]"))
            else:
                stored.append(("text", "[image omitted]"))
    return stored


def store_incoming_email(
    *,
    conn,
    incoming: IncomingEmail,
    whitelist: set[str] | list[str],
    media_root: Path | None = None,
) -> dict:
    sender = normalize_sender(incoming.from_header)
    normalized_whitelist = {s.strip().lower() for s in whitelist}
    if not sender or sender not in normalized_whitelist:
        return {"status": "ignored_non_whitelisted_sender"}

    incoming_message_id = normalize_message_id(incoming.message_id)
    in_reply_to = normalize_message_id(incoming.in_reply_to)
    references = normalize_references(incoming.references)

    if incoming_message_id and db.has_incoming_email_message_id(conn, incoming_message_id):
        return {"status": "ignored_duplicate_message"}

    thread_id = db.resolve_email_thread_from_references(
        conn,
        sender_id=sender,
        in_reply_to=in_reply_to,
        references=references,
    )
    created_ts = db.convert_to_local_iso(incoming.timestamp)
    if thread_id is None:
        thread_id = db.create_email_thread(conn, sender_id=sender, created_ts=created_ts)

    content_parts = _content_parts_for_storage(incoming, media_root)
    if not content_parts:
        content_parts = [("text", "")]

    message_row_id = db.insert_message_with_content(
        conn,
        channel="email",
        thread_id=str(thread_id),
        sender_id=sender,
        role="user",
        timestamp=created_ts,
        content_parts=content_parts,
    )
    db.insert_email_message_meta(
        conn,
        message_id=message_row_id,
        email_message_id=incoming_message_id,
        in_reply_to=in_reply_to,
        references_header=" ".join(references) if references else None,
        subject=incoming.subject,
        process_state=db.PENDING,
    )

    return {
        "status": "queued",
        "sender": sender,
        "email_thread_id": thread_id,
        "incoming_message_id": incoming_message_id,
        "message_id": message_row_id,
    }


def build_reply_email(
    *,
    from_address: str,
    incoming: IncomingEmail,
    to: str,
    body: str,
    generated_message_id: str | None = None,
) -> tuple[OutgoingEmail, str]:
    incoming_message_id = normalize_message_id(incoming.message_id)
    references = normalize_references(incoming.references)
    outgoing_message_id, headers = _build_outgoing_headers(
        incoming_message_id=incoming_message_id,
        references=references,
        generated_message_id=generated_message_id,
    )
    return (
        OutgoingEmail(
            to=to,
            from_address=from_address,
            subject=_subject_for_reply(incoming.subject),
            body=body,
            headers=headers,
        ),
        outgoing_message_id,
    )


def send_reply_smtp(outgoing: OutgoingEmail, smtp_config: SMTPConfigProtocol) -> None:
    message = EmailMessage()
    message["From"] = formataddr(("PodHead", outgoing.from_address))
    message["To"] = outgoing.to
    message["Subject"] = outgoing.subject
    for key, value in outgoing.headers.items():
        message[key] = value
    message.set_content(outgoing.body)

    try:
        with smtplib.SMTP(smtp_config.host, smtp_config.port, timeout=30) as smtp:
            if smtp_config.use_tls:
                smtp.starttls()
            smtp.login(smtp_config.username, smtp_config.password)
            smtp.send_message(message)
    except Exception as exc:  # noqa: BLE001
        raise SMTPDeliveryError(str(exc)) from exc


def _open_imap_connection(imap_config: IMAPConfigProtocol):
    if imap_config.use_ssl:
        return imaplib.IMAP4_SSL(imap_config.host, imap_config.port)
    return imaplib.IMAP4(imap_config.host, imap_config.port)


def poll_inbox(
    *,
    conn,
    whitelist: set[str] | list[str],
    imap_config: IMAPConfigProtocol,
    media_root: Path | None = None,
) -> list[dict]:
    try:
        with _open_imap_connection(imap_config) as client:
            client.login(imap_config.username, imap_config.password)
            status, _ = client.select(imap_config.inbox)
            if status != "OK":
                raise IMAPPollError(f"Failed to select inbox {imap_config.inbox!r}")

            status, data = client.search(None, "ALL")
            if status != "OK":
                raise IMAPPollError("Failed to search inbox")

            results: list[dict] = []
            for raw_id in data[0].split():
                status, fetched = client.fetch(raw_id, "(RFC822)")
                if status != "OK" or not fetched or fetched[0] is None:
                    continue
                payload = fetched[0][1]
                if not isinstance(payload, (bytes, bytearray)):
                    continue
                incoming = parse_mime_message(bytes(payload))
                result = store_incoming_email(
                    conn=conn,
                    incoming=incoming,
                    whitelist=whitelist,
                    media_root=media_root,
                )
                results.append(result)
                client.store(raw_id, "+FLAGS", "\\Seen")
            return results
    except IMAPPollError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise IMAPPollError(str(exc)) from exc


def process_incoming_email(
    *,
    conn,
    incoming: IncomingEmail,
    whitelist: set[str] | list[str],
    run_agent: Callable[[list[dict], dict], str],
    send_reply: Callable[[OutgoingEmail], None],
    generated_message_id: str | None = None,
    from_address: str = "podhead@example.com",
    media_root: Path | None = None,
) -> dict:
    """Store, run agent, send reply, store reply."""
    result = store_incoming_email(
        conn=conn, incoming=incoming, whitelist=whitelist, media_root=media_root
    )
    if result["status"] != "queued":
        return result

    thread_id = str(result["email_thread_id"])
    history = db.get_conversation(conn, "email", thread_id)
    message_id = result["message_id"]
    db.update_email_message_state(conn, message_id, db.PROCESSING)

    current_message = next(m for m in history if m["id"] == message_id)

    try:
        assistant_text = run_agent(history, current_message)

        meta = db.get_email_message_meta(conn, message_id)
        temp_incoming = IncomingEmail(
            from_header=result["sender"],
            subject=(meta.get("subject") if meta else None) or incoming.subject,
            content_parts=[],
            message_id=meta.get("email_message_id") if meta else None,
            in_reply_to=meta.get("in_reply_to") if meta else None,
            references=meta.get("references_header") if meta else None,
        )
        outgoing, outgoing_message_id = build_reply_email(
            from_address=from_address,
            incoming=temp_incoming,
            to=result["sender"],
            body=assistant_text,
            generated_message_id=generated_message_id,
        )
        send_reply(outgoing)

        ts = db.now_local_iso()
        asst_message_id = db.insert_message_with_content(
            conn,
            channel="email",
            thread_id=thread_id,
            sender_id="assistant",
            role="assistant",
            timestamp=ts,
            content_parts=[("text", assistant_text)],
        )
        db.insert_email_message_meta(
            conn,
            message_id=asst_message_id,
            email_message_id=outgoing_message_id,
            subject=outgoing.subject,
            process_state=db.COMPLETED,
        )
        db.update_email_message_state(conn, message_id, db.COMPLETED)
    except Exception as exc:  # noqa: BLE001
        db.update_email_message_state(conn, message_id, db.FAILED, str(exc))
        raise

    result["status"] = "processed"
    result["outgoing_message_id"] = outgoing_message_id
    return result
