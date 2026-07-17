"""SQLite schema and access helpers for conversation and message storage."""

from __future__ import annotations

import sqlite3
import time
from typing import Iterable


def now_ts() -> int:
    """Return current unix timestamp (seconds)."""
    return int(time.time())


def init_db(conn: sqlite3.Connection) -> None:
    """Create tables and indexes required by mail-thread routing."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender_normalized TEXT NOT NULL,
            created_ts INTEGER NOT NULL,
            last_activity_ts INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            email_message_id TEXT UNIQUE,
            direction TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            in_reply_to TEXT,
            references_header TEXT,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_messages_conversation_time
        ON messages(conversation_id, timestamp, id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_messages_email_message_id
        ON messages(email_message_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_conversations_sender
        ON conversations(sender_normalized)
        """
    )
    conn.commit()


def create_conversation(
    conn: sqlite3.Connection,
    sender_normalized: str,
    created_ts: int | None = None,
) -> int:
    """Create and return a new conversation id."""
    ts = now_ts() if created_ts is None else created_ts
    cur = conn.execute(
        """
        INSERT INTO conversations(sender_normalized, created_ts, last_activity_ts)
        VALUES(?, ?, ?)
        """,
        (sender_normalized, ts, ts),
    )
    conn.commit()
    return int(cur.lastrowid)


def touch_conversation(
    conn: sqlite3.Connection,
    conversation_id: int,
    activity_ts: int | None = None,
) -> None:
    """Update last activity timestamp for a conversation."""
    ts = now_ts() if activity_ts is None else activity_ts
    conn.execute(
        "UPDATE conversations SET last_activity_ts=? WHERE id=?",
        (ts, conversation_id),
    )
    conn.commit()


def insert_message(
    conn: sqlite3.Connection,
    *,
    conversation_id: int,
    email_message_id: str | None,
    direction: str,
    content: str,
    timestamp: int | None = None,
    in_reply_to: str | None = None,
    references_header: str | None = None,
) -> int:
    """Insert a message row and return the internal message id."""
    ts = now_ts() if timestamp is None else timestamp
    cur = conn.execute(
        """
        INSERT INTO messages(
            conversation_id,
            email_message_id,
            direction,
            content,
            timestamp,
            in_reply_to,
            references_header
        )
        VALUES(?, ?, ?, ?, ?, ?, ?)
        """,
        (
            conversation_id,
            email_message_id,
            direction,
            content,
            ts,
            in_reply_to,
            references_header,
        ),
    )
    touch_conversation(conn, conversation_id, ts)
    return int(cur.lastrowid)


def get_conversation_sender(
    conn: sqlite3.Connection,
    conversation_id: int,
) -> str | None:
    """Return normalized sender for a conversation."""
    row = conn.execute(
        "SELECT sender_normalized FROM conversations WHERE id=?",
        (conversation_id,),
    ).fetchone()
    if row is None:
        return None
    return str(row[0])


def get_conversation_by_email_message_id(
    conn: sqlite3.Connection,
    email_message_id: str,
) -> int | None:
    """Return conversation id for an email message id, if found."""
    row = conn.execute(
        """
        SELECT conversation_id
        FROM messages
        WHERE email_message_id=?
        LIMIT 1
        """,
        (email_message_id,),
    ).fetchone()
    if row is None:
        return None
    return int(row[0])


def resolve_conversation_from_references(
    conn: sqlite3.Connection,
    *,
    sender_normalized: str,
    in_reply_to: str | None,
    references: Iterable[str],
) -> int | None:
    """Resolve conversation id by reply headers, scoped to sender ownership."""
    # RFC References arrive oldest->newest; route preference is newest->oldest.
    refs_newest_first = list(reversed(references))
    candidates: list[str] = []
    if in_reply_to:
        candidates.append(in_reply_to)
    for message_id in refs_newest_first:
        candidates.append(message_id)

    for message_id in candidates:
        conversation_id = get_conversation_by_email_message_id(conn, message_id)
        if conversation_id is None:
            continue
        owner = get_conversation_sender(conn, conversation_id)
        if owner == sender_normalized:
            return conversation_id
    return None


def has_incoming_message_id(conn: sqlite3.Connection, email_message_id: str) -> bool:
    """Return True when an incoming email with this Message-ID is already stored."""
    row = conn.execute(
        """
        SELECT 1
        FROM messages
        WHERE email_message_id=? AND direction='incoming'
        LIMIT 1
        """,
        (email_message_id,),
    ).fetchone()
    return row is not None


def get_conversation_history(
    conn: sqlite3.Connection,
    conversation_id: int,
) -> list[dict]:
    """Return all messages for one conversation, ordered oldest first."""
    rows = conn.execute(
        """
        SELECT
            id,
            conversation_id,
            email_message_id,
            direction,
            content,
            timestamp,
            in_reply_to,
            references_header
        FROM messages
        WHERE conversation_id=?
        ORDER BY timestamp ASC, id ASC
        """,
        (conversation_id,),
    ).fetchall()
    return [
        {
            "id": int(r[0]),
            "conversation_id": int(r[1]),
            "email_message_id": r[2],
            "direction": r[3],
            "content": r[4],
            "timestamp": int(r[5]),
            "in_reply_to": r[6],
            "references_header": r[7],
        }
        for r in rows
    ]
