"""Mail parsing, routing, IMAP polling, and SMTP sending helpers."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from email import message_from_bytes, policy
from email.message import EmailMessage, Message
from email.utils import formataddr, make_msgid, parseaddr, parsedate_to_datetime
from html.parser import HTMLParser
import hashlib
import imaplib
import json
from pathlib import Path
import re
import secrets
import smtplib
from typing import NamedTuple

from backhead import db
from backhead import media as media_mod

MESSAGE_ID_PATTERN = re.compile(r"<[^<>]+>")
THREAD_MARKER_PATTERN = re.compile(r"podhead\.([0-9a-f]{64})\.", re.IGNORECASE)


class ContentPart(NamedTuple):
    kind: str
    text: str = ""
    image_bytes: bytes = b""
    mime_type: str = ""


@dataclass(frozen=True)
class IncomingEmail:
    from_header: str
    subject: str
    content_parts: list
    message_id: str | None = None
    in_reply_to: str | None = None
    references: str | None = None
    timestamp: int | float | None = None

    @property
    def body(self) -> str:
        texts = [p.text for p in self.content_parts if p.kind == "text"]
        return "\n\n".join(t for t in texts if t)


@dataclass(frozen=True)
class OutgoingEmail:
    to: str
    from_address: str
    subject: str
    body: str
    headers: dict[str, str]


@dataclass(frozen=True)
class EmailTransportData:
    imap_identifier: str | None
    incoming_message_id: str | None
    in_reply_to: str | None
    references: str | None
    subject: str
    sender_id: str
    thread_id: str


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


def deterministic_thread_id(sender_id: str, incoming_message_id: str) -> str:
    source = json.dumps(
        {"sender_id": sender_id, "incoming_message_id": incoming_message_id},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(source).hexdigest()


def decode_thread_id_from_message_id(message_id: str | None) -> str | None:
    if not message_id:
        return None
    match = THREAD_MARKER_PATTERN.search(message_id)
    if not match:
        return None
    return match.group(1).lower()


def resolve_thread_id(
    *,
    sender_id: str,
    incoming_message_id: str | None,
    in_reply_to: str | None,
    references: list[str],
) -> str:
    candidates: list[str] = []
    if in_reply_to:
        candidates.append(in_reply_to)
    candidates.extend(reversed(references))
    for candidate in candidates:
        decoded = decode_thread_id_from_message_id(candidate)
        if decoded:
            return decoded
    if incoming_message_id:
        return deterministic_thread_id(sender_id, incoming_message_id)
    return secrets.token_hex(32)


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


def _content_id(part: Message) -> str | None:
    cid = part.get("Content-ID")
    if not cid:
        return None
    return cid.strip().strip("<>").lower()


def parse_mime_message(raw_message: bytes) -> IncomingEmail:
    parsed = message_from_bytes(raw_message, policy=policy.default)
    parts = _extract_content_parts(parsed)
    date_str = parsed.get("Date")
    timestamp = None
    if date_str:
        try:
            timestamp = parsedate_to_datetime(date_str.strip()).timestamp()
        except (TypeError, ValueError):
            timestamp = None
    return IncomingEmail(
        from_header=parsed.get("From", ""),
        subject=parsed.get("Subject", ""),
        content_parts=parts,
        message_id=parsed.get("Message-ID"),
        in_reply_to=parsed.get("In-Reply-To"),
        references=parsed.get("References"),
        timestamp=timestamp,
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
                parts.append(ContentPart(kind="image_bytes", image_bytes=payload, mime_type=ctype))
        elif maintype == "audio":
            parts.append(ContentPart(kind="text", text="Audio detected. STT not yet implemented"))
        elif maintype == "video":
            parts.append(ContentPart(kind="text", text="Video detected. Video normalization not yet implemented"))

    if not have_plain_text and html_fallback:
        parts.insert(0, ContentPart(kind="text", text=html_fallback))

    return parts


def _subject_for_reply(subject: str) -> str:
    if not subject:
        return "Re: (no subject)"
    if subject.lower().strip().startswith("re:"):
        return subject
    return f"Re: {subject}"


def _build_references_header(existing_refs: list[str], incoming_message_id: str | None) -> str | None:
    refs = list(existing_refs)
    if incoming_message_id and incoming_message_id not in refs:
        refs.append(incoming_message_id)
    if not refs:
        return None
    return " ".join(refs)


def generate_outgoing_message_id(thread_id: str) -> str:
    token = secrets.token_hex(8)
    generated = normalize_message_id(make_msgid(idstring=f"podhead.{thread_id}.{token}"))
    if generated is None:
        raise ValueError("Failed to generate a valid Message-ID for outgoing email")
    return generated


def build_reply_email(
    *,
    from_address: str,
    to: str,
    subject: str,
    body: str,
    thread_id: str,
    incoming_message_id: str | None,
    references_header: str | None,
) -> tuple[OutgoingEmail, str]:
    references = normalize_references(references_header)
    outgoing_message_id = generate_outgoing_message_id(thread_id)
    headers: dict[str, str] = {"Message-ID": outgoing_message_id}
    if incoming_message_id:
        headers["In-Reply-To"] = incoming_message_id
    references_out = _build_references_header(references, incoming_message_id)
    if references_out:
        headers["References"] = references_out

    return (
        OutgoingEmail(
            to=to,
            from_address=from_address,
            subject=_subject_for_reply(subject),
            body=body,
            headers=headers,
        ),
        outgoing_message_id,
    )


def _content_parts_for_storage(incoming: IncomingEmail, media_root: Path | None) -> list[tuple[str, str]]:
    stored: list[tuple[str, str]] = []
    for part in incoming.content_parts:
        if part.kind == "text":
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
    imap_identifier: str | None = None,
) -> dict:
    sender = normalize_sender(incoming.from_header)
    normalized_whitelist = {s.strip().lower() for s in whitelist}
    if not sender or sender not in normalized_whitelist:
        return {"status": "ignored_non_whitelisted_sender"}

    incoming_message_id = normalize_message_id(incoming.message_id)
    in_reply_to = normalize_message_id(incoming.in_reply_to)
    references = normalize_references(incoming.references)
    thread_id = resolve_thread_id(
        sender_id=sender,
        incoming_message_id=incoming_message_id,
        in_reply_to=in_reply_to,
        references=references,
    )
    created_ts = db.convert_to_local_iso(incoming.timestamp)

    content_parts = _content_parts_for_storage(incoming, media_root)
    if not content_parts:
        content_parts = [("text", "")]

    message_row_id = db.insert_message_with_content(
        conn,
        channel="email",
        thread_id=thread_id,
        sender_id=sender,
        role="user",
        timestamp=created_ts,
        content_parts=content_parts,
    )

    transport = EmailTransportData(
        imap_identifier=imap_identifier,
        incoming_message_id=incoming_message_id,
        in_reply_to=in_reply_to,
        references=" ".join(references) if references else None,
        subject=incoming.subject,
        sender_id=sender,
        thread_id=thread_id,
    )
    return {
        "status": "queued",
        "sender": sender,
        "thread_id": thread_id,
        "incoming_message_id": incoming_message_id,
        "message_id": message_row_id,
        "transport": transport,
    }


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


def _extract_response_bytes(response) -> bytes | None:
    if not response:
        return None
    for item in response:
        if isinstance(item, tuple) and len(item) > 1 and isinstance(item[1], (bytes, bytearray)):
            return bytes(item[1])
    return None


def _fetch_uid_bytes(client, uid: bytes, fetch_spec: str) -> bytes | None:
    status, fetched = client.uid("FETCH", uid, fetch_spec)
    if status != "OK":
        return None
    return _extract_response_bytes(fetched)


def _move_to_mailbox(client, uid: bytes, mailbox: str) -> None:
    status, _ = client.uid("MOVE", uid, mailbox)
    if status == "OK":
        return
    status, _ = client.uid("COPY", uid, mailbox)
    if status != "OK":
        raise IMAPPollError(f"Failed to move message to mailbox {mailbox!r}")
    status, _ = client.uid("STORE", uid, "+FLAGS.SILENT", r"(\\Deleted)")
    if status != "OK":
        raise IMAPPollError("Failed to mark source message deleted after copy fallback")
    client.expunge()


def _sender_from_header_bytes(header_bytes: bytes) -> str | None:
    parsed = message_from_bytes(header_bytes, policy=policy.default)
    return normalize_sender(parsed.get("From", ""))


def poll_inbox(
    *,
    conn,
    whitelist: set[str] | list[str],
    imap_config: IMAPConfigProtocol,
    spam_mailbox: str,
    media_root: Path | None = None,
    db_lock=None,
) -> list[dict]:
    _lock = db_lock if db_lock is not None else contextlib.nullcontext()
    normalized_whitelist = {s.strip().lower() for s in whitelist}
    try:
        with _open_imap_connection(imap_config) as client:
            client.login(imap_config.username, imap_config.password)
            status, _ = client.select(imap_config.inbox)
            if status != "OK":
                raise IMAPPollError(f"Failed to select inbox {imap_config.inbox!r}")

            status, data = client.uid("SEARCH", None, "UNSEEN")
            if status != "OK":
                raise IMAPPollError("Failed to search inbox")

            results: list[dict] = []
            for uid in data[0].split():
                header_bytes = _fetch_uid_bytes(client, uid, "(BODY.PEEK[HEADER.FIELDS (FROM)])")
                if not header_bytes:
                    continue
                sender = _sender_from_header_bytes(header_bytes)
                if not sender or sender not in normalized_whitelist:
                    _move_to_mailbox(client, uid, spam_mailbox)
                    results.append({"status": "moved_to_spam", "sender": sender})
                    continue

                payload = _fetch_uid_bytes(client, uid, "(BODY.PEEK[])")
                if not isinstance(payload, (bytes, bytearray)):
                    continue
                incoming = parse_mime_message(bytes(payload))

                # Prepare all data outside the lock
                incoming_message_id = normalize_message_id(incoming.message_id)
                in_reply_to = normalize_message_id(incoming.in_reply_to)
                references = normalize_references(incoming.references)
                thread_id = resolve_thread_id(
                    sender_id=sender,
                    incoming_message_id=incoming_message_id,
                    in_reply_to=in_reply_to,
                    references=references,
                )
                created_ts = db.convert_to_local_iso(incoming.timestamp)
                content_parts = _content_parts_for_storage(incoming, media_root)
                if not content_parts:
                    content_parts = [("text", "")]
                uid_str = uid.decode("ascii", errors="ignore")

                # Acquire lock only around the DB insertion
                with _lock:
                    message_row_id = db.insert_message_with_content(
                        conn,
                        channel="email",
                        thread_id=thread_id,
                        sender_id=sender,
                        role="user",
                        timestamp=created_ts,
                        content_parts=content_parts,
                    )

                # Mark as Seen only after successful storage
                client.uid("STORE", uid, "+FLAGS.SILENT", r"(\\Seen)")

                transport = EmailTransportData(
                    imap_identifier=uid_str,
                    incoming_message_id=incoming_message_id,
                    in_reply_to=in_reply_to,
                    references=" ".join(references) if references else None,
                    subject=incoming.subject,
                    sender_id=sender,
                    thread_id=thread_id,
                )
                results.append({
                    "status": "queued",
                    "sender": sender,
                    "thread_id": thread_id,
                    "incoming_message_id": incoming_message_id,
                    "message_id": message_row_id,
                    "transport": transport,
                })
            return results
    except IMAPPollError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise IMAPPollError(str(exc)) from exc
