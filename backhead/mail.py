"""Mail parsing and conversation routing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from email.utils import make_msgid, parseaddr
import re
from typing import Callable

from backhead import db

MESSAGE_ID_PATTERN = re.compile(r"<[^<>]+>")


@dataclass(frozen=True)
class IncomingEmail:
    """Canonical incoming email payload used by the backend."""

    from_header: str
    subject: str
    body: str
    message_id: str | None = None
    in_reply_to: str | None = None
    references: str | None = None
    timestamp: int | None = None


@dataclass(frozen=True)
class OutgoingEmail:
    """Outgoing reply payload that can be delivered by SMTP backend."""

    to: str
    subject: str
    body: str
    headers: dict[str, str]


def normalize_sender(from_header: str) -> str | None:
    """Normalize sender using parsed address from From header."""
    _, address = parseaddr(from_header or "")
    normalized = (address or "").strip().lower()
    return normalized or None


def normalize_message_id(value: str | None) -> str | None:
    """Normalize Message-ID/In-Reply-To values into canonical '<id@host>' form."""
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
    """Parse and normalize References header into oldest->newest message ids."""
    if not value:
        return []
    refs = [m.group(0).strip().lower() for m in MESSAGE_ID_PATTERN.finditer(value)]
    return refs


def _subject_for_reply(subject: str) -> str:
    if not subject:
        return "Re: (no subject)"
    lowered = subject.lower().strip()
    if lowered.startswith("re:"):
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


def process_incoming_email(
    *,
    conn,
    incoming: IncomingEmail,
    whitelist: set[str] | list[str],
    run_agent: Callable[[list[dict], IncomingEmail], str],
    send_reply: Callable[[OutgoingEmail], None],
    generated_message_id: str | None = None,
) -> dict:
    """Process one incoming email into one routed conversation turn."""
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

    db.insert_message(
        conn,
        conversation_id=conversation_id,
        email_message_id=incoming_message_id,
        direction="incoming",
        content=incoming.body,
        timestamp=incoming.timestamp,
        in_reply_to=in_reply_to,
        references_header=" ".join(references) if references else None,
    )

    history = db.get_conversation_history(conn, conversation_id)
    assistant_text = run_agent(history, incoming)

    outgoing_message_id, headers = _build_outgoing_headers(
        incoming_message_id=incoming_message_id,
        references=references,
        generated_message_id=generated_message_id,
    )

    db.insert_message(
        conn,
        conversation_id=conversation_id,
        email_message_id=outgoing_message_id,
        direction="outgoing",
        content=assistant_text,
    )

    send_reply(
        OutgoingEmail(
            to=sender,
            subject=_subject_for_reply(incoming.subject),
            body=assistant_text,
            headers=headers,
        )
    )
    return {
        "status": "processed",
        "sender": sender,
        "conversation_id": conversation_id,
        "incoming_message_id": incoming_message_id,
        "outgoing_message_id": outgoing_message_id,
    }
