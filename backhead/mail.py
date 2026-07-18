"""Mail parsing, routing, IMAP polling, and SMTP sending helpers."""

from __future__ import annotations

from dataclasses import dataclass
from email import message_from_bytes, policy
from email.message import EmailMessage, Message
from email.utils import formataddr, make_msgid, parseaddr
import imaplib
import re
import smtplib
from typing import Callable

from backhead import db

MESSAGE_ID_PATTERN = re.compile(r"<[^<>]+>")


@dataclass(frozen=True)
class IncomingEmail:
    from_header: str
    subject: str
    body: str
    message_id: str | None = None
    in_reply_to: str | None = None
    references: str | None = None
    timestamp: int | None = None


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



def parse_mime_message(raw_message: bytes) -> IncomingEmail:
    parsed = message_from_bytes(raw_message, policy=policy.default)
    return IncomingEmail(
        from_header=parsed.get("From", ""),
        subject=parsed.get("Subject", ""),
        body=_extract_text_body(parsed),
        message_id=parsed.get("Message-ID"),
        in_reply_to=parsed.get("In-Reply-To"),
        references=parsed.get("References"),
        timestamp=None,
    )



def _extract_text_body(message: Message) -> str:
    if message.is_multipart():
        text_parts: list[str] = []
        for part in message.walk():
            if part.get_content_maintype() == "multipart":
                continue
            if part.get_content_disposition() == "attachment":
                continue
            if part.get_content_type() == "text/plain":
                text_parts.append(part.get_content())
        if text_parts:
            return "\n\n".join(part.strip() for part in text_parts if part.strip())
        for part in message.walk():
            if part.get_content_type() == "text/html":
                return part.get_content().strip()
        return ""
    content = message.get_content()
    return content.strip() if isinstance(content, str) else ""



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



def store_incoming_email(
    *,
    conn,
    incoming: IncomingEmail,
    whitelist: set[str] | list[str],
) -> dict:
    sender = normalize_sender(incoming.from_header)
    normalized_whitelist = {s.strip().lower() for s in whitelist}
    if not sender or sender not in normalized_whitelist:
        return {"status": "ignored_non_whitelisted_sender"}

    incoming_message_id = normalize_message_id(incoming.message_id)
    in_reply_to = normalize_message_id(incoming.in_reply_to)
    references = normalize_references(incoming.references)

    if incoming_message_id and db.has_incoming_message_id(conn, incoming_message_id):
        return {"status": "ignored_duplicate_message"}

    conversation_id = db.resolve_conversation_from_references(
        conn,
        sender_normalized=sender,
        in_reply_to=in_reply_to,
        references=references,
    )
    if conversation_id is None:
        conversation_id = db.create_conversation(
            conn,
            sender_normalized=sender,
            created_ts=incoming.timestamp,
        )

    message_row_id = db.insert_message(
        conn,
        conversation_id=conversation_id,
        email_message_id=incoming_message_id,
        direction="incoming",
        content=incoming.body,
        subject=incoming.subject,
        timestamp=incoming.timestamp,
        in_reply_to=in_reply_to,
        references_header=" ".join(references) if references else None,
        process_state=db.PENDING,
    )

    return {
        "status": "queued",
        "sender": sender,
        "conversation_id": conversation_id,
        "incoming_message_id": incoming_message_id,
        "message_row_id": message_row_id,
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



def poll_inbox(*, conn, whitelist: set[str] | list[str], imap_config: IMAPConfigProtocol) -> list[dict]:
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
                result = store_incoming_email(conn=conn, incoming=incoming, whitelist=whitelist)
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
    run_agent: Callable[[list[dict], IncomingEmail], str],
    send_reply: Callable[[OutgoingEmail], None],
    generated_message_id: str | None = None,
    from_address: str = "podhead@example.com",
) -> dict:
    result = store_incoming_email(conn=conn, incoming=incoming, whitelist=whitelist)
    if result["status"] != "queued":
        return result

    history = db.get_conversation_history(conn, result["conversation_id"])
    db.update_message_state(conn, result["message_row_id"], db.PROCESSING)
    try:
        assistant_text = run_agent(history, incoming)
        outgoing, outgoing_message_id = build_reply_email(
            from_address=from_address,
            incoming=incoming,
            to=result["sender"],
            body=assistant_text,
            generated_message_id=generated_message_id,
        )
        send_reply(outgoing)
        db.insert_message(
            conn,
            conversation_id=result["conversation_id"],
            email_message_id=outgoing_message_id,
            direction="outgoing",
            content=assistant_text,
            process_state=db.COMPLETED,
        )
        db.update_message_state(conn, result["message_row_id"], db.COMPLETED)
    except Exception as exc:  # noqa: BLE001
        db.update_message_state(conn, result["message_row_id"], db.FAILED, str(exc))
        raise

    result["status"] = "processed"
    result["outgoing_message_id"] = outgoing_message_id
    return result
