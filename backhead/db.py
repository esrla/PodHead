"""SQLite schema and access helpers for conversation and message storage."""

from __future__ import annotations

import sqlite3
import time
from typing import Iterable

PENDING = "pending"
PROCESSING = "processing"
COMPLETED = "completed"
FAILED = "failed"
INCOMING_STATES = {PENDING, PROCESSING, COMPLETED, FAILED}


def now_ts() -> int:
    return int(time.time())


def _row_to_message(row: sqlite3.Row | tuple) -> dict:
    return {
        "id": int(row[0]),
        "conversation_id": int(row[1]),
        "email_message_id": row[2],
        "direction": row[3],
        "content": row[4],
        "subject": row[5],
        "timestamp": int(row[6]),
        "in_reply_to": row[7],
        "references_header": row[8],
        "process_state": row[9],
        "failure_details": row[10],
    }


def init_db(conn: sqlite3.Connection) -> None:
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
            subject TEXT,
            timestamp INTEGER NOT NULL,
            in_reply_to TEXT,
            references_header TEXT,
            process_state TEXT NOT NULL DEFAULT 'completed',
            failure_details TEXT,
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
        CREATE INDEX IF NOT EXISTS idx_messages_state_time
        ON messages(process_state, conversation_id, timestamp, id)
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
    subject: str | None = None,
    in_reply_to: str | None = None,
    references_header: str | None = None,
    process_state: str = COMPLETED,
    failure_details: str | None = None,
) -> int:
    ts = now_ts() if timestamp is None else timestamp
    cur = conn.execute(
        """
        INSERT INTO messages(
            conversation_id,
            email_message_id,
            direction,
            content,
            subject,
            timestamp,
            in_reply_to,
            references_header,
            process_state,
            failure_details
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            conversation_id,
            email_message_id,
            direction,
            content,
            subject,
            ts,
            in_reply_to,
            references_header,
            process_state,
            failure_details,
        ),
    )
    touch_conversation(conn, conversation_id, ts)
    return int(cur.lastrowid)


def update_message_state(
    conn: sqlite3.Connection,
    message_id: int,
    process_state: str,
    failure_details: str | None = None,
) -> None:
    conn.execute(
        "UPDATE messages SET process_state=?, failure_details=? WHERE id=?",
        (process_state, failure_details, message_id),
    )
    conn.commit()


def get_message(conn: sqlite3.Connection, message_id: int) -> dict | None:
    row = conn.execute(
        """
        SELECT id, conversation_id, email_message_id, direction, content, subject, timestamp,
               in_reply_to, references_header, process_state, failure_details
        FROM messages
        WHERE id=?
        """,
        (message_id,),
    ).fetchone()
    if row is None:
        return None
    return _row_to_message(row)


def get_conversation_sender(conn: sqlite3.Connection, conversation_id: int) -> str | None:
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
    refs_newest_first = list(reversed(list(references)))
    candidates: list[str] = []
    if in_reply_to:
        candidates.append(in_reply_to)
    candidates.extend(refs_newest_first)
    for message_id in candidates:
        conversation_id = get_conversation_by_email_message_id(conn, message_id)
        if conversation_id is None:
            continue
        if get_conversation_sender(conn, conversation_id) == sender_normalized:
            return conversation_id
    return None


def has_incoming_message_id(conn: sqlite3.Connection, email_message_id: str) -> bool:
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


def get_conversation_history(conn: sqlite3.Connection, conversation_id: int) -> list[dict]:
    rows = conn.execute(
        """
        SELECT id, conversation_id, email_message_id, direction, content, subject, timestamp,
               in_reply_to, references_header, process_state, failure_details
        FROM messages
        WHERE conversation_id=?
        ORDER BY timestamp ASC, id ASC
        """,
        (conversation_id,),
    ).fetchall()
    return [_row_to_message(row) for row in rows]


def reset_processing_messages(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        UPDATE messages
        SET process_state='pending', failure_details=NULL
        WHERE direction='incoming' AND process_state='processing'
        """
    )
    conn.commit()


def requeue_failed_messages(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        UPDATE messages
        SET process_state='pending'
        WHERE direction='incoming' AND process_state='failed'
        """
    )
    conn.commit()


def list_conversations_with_work(conn: sqlite3.Connection) -> list[int]:
    rows = conn.execute(
        """
        SELECT DISTINCT conversation_id
        FROM messages
        WHERE direction='incoming' AND process_state='pending'
        ORDER BY conversation_id ASC
        """
    ).fetchall()
    return [int(row[0]) for row in rows]


def get_next_pending_message(
    conn: sqlite3.Connection,
    conversation_id: int,
) -> dict | None:
    row = conn.execute(
        """
        SELECT id, conversation_id, email_message_id, direction, content, subject, timestamp,
               in_reply_to, references_header, process_state, failure_details
        FROM messages
        WHERE conversation_id=? AND direction='incoming' AND process_state='pending'
        ORDER BY timestamp ASC, id ASC
        LIMIT 1
        """,
        (conversation_id,),
    ).fetchone()
    if row is None:
        return None
    return _row_to_message(row)


def claim_next_pending_message(
    conn: sqlite3.Connection,
    conversation_id: int,
) -> dict | None:
    row = get_next_pending_message(conn, conversation_id)
    if row is None:
        return None
    updated = conn.execute(
        """
        UPDATE messages
        SET process_state='processing', failure_details=NULL
        WHERE id=? AND process_state='pending'
        """,
        (row["id"],),
    )
    conn.commit()
    if updated.rowcount != 1:
        return None
    row["process_state"] = PROCESSING
    row["failure_details"] = None
    return row
