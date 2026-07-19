"""SQLite schema and access helpers for the generic conversation log.

Two generic tables (``messages`` + ``message_content``) hold channel-agnostic
conversation history. Two email-specific tables (``email_threads`` +
``email_message_meta``) hold email threading and processing state.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

PENDING = "pending"
PROCESSING = "processing"
COMPLETED = "completed"
FAILED = "failed"
INCOMING_STATES = {PENDING, PROCESSING, COMPLETED, FAILED}


def now_local_iso() -> str:
    """Return current time in the backend's local timezone as ISO 8601 with offset."""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def convert_to_local_iso(ts: int | float | None) -> str:
    """Convert a Unix timestamp (or ``None`` -> now) to local timezone ISO 8601."""
    if ts is None:
        return now_local_iso()
    return datetime.fromtimestamp(ts).astimezone().isoformat(timespec="milliseconds")


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel TEXT NOT NULL,
            thread_id TEXT NOT NULL,
            sender_id TEXT NOT NULL,
            role TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS message_content (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id INTEGER NOT NULL,
            ordinal INTEGER NOT NULL,
            content_type TEXT NOT NULL,
            content TEXT NOT NULL,
            UNIQUE(message_id, ordinal),
            FOREIGN KEY(message_id) REFERENCES messages(id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS email_threads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender_id TEXT NOT NULL,
            created_ts TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS email_message_meta (
            message_id INTEGER PRIMARY KEY,
            email_message_id TEXT UNIQUE,
            in_reply_to TEXT,
            references_header TEXT,
            subject TEXT,
            process_state TEXT NOT NULL DEFAULT 'completed',
            failure_details TEXT,
            FOREIGN KEY(message_id) REFERENCES messages(id)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_messages_channel_thread
        ON messages(channel, thread_id, id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_message_content_message
        ON message_content(message_id, ordinal)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_email_message_meta_email_id
        ON email_message_meta(email_message_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_email_message_meta_state
        ON email_message_meta(process_state)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_email_threads_sender
        ON email_threads(sender_id)
        """
    )
    conn.commit()


# --------------------------------------------------------------------------- #
# Generic message functions
# --------------------------------------------------------------------------- #


def insert_message_with_content(
    conn: sqlite3.Connection,
    *,
    channel: str,
    thread_id: str,
    sender_id: str,
    role: str,
    timestamp: str,
    content_parts: list[tuple[str, str]],
) -> int:
    """Atomically insert a message and all its content rows. Returns messages.id.

    ``content_parts`` is ``[(content_type, content), ...]`` where the list index
    becomes the content ordinal. If any insert fails, the whole operation is
    rolled back and nothing is stored.
    """
    try:
        cur = conn.execute(
            """
            INSERT INTO messages(channel, thread_id, sender_id, role, timestamp)
            VALUES(?, ?, ?, ?, ?)
            """,
            (channel, thread_id, sender_id, role, timestamp),
        )
        message_id = int(cur.lastrowid)
        for ordinal, (content_type, content) in enumerate(content_parts):
            conn.execute(
                """
                INSERT INTO message_content(message_id, ordinal, content_type, content)
                VALUES(?, ?, ?, ?)
                """,
                (message_id, ordinal, content_type, content),
            )
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return message_id


def _load_content(conn: sqlite3.Connection, message_id: int) -> list[dict]:
    rows = conn.execute(
        """
        SELECT ordinal, content_type, content
        FROM message_content
        WHERE message_id=?
        ORDER BY ordinal ASC
        """,
        (message_id,),
    ).fetchall()
    return [
        {"ordinal": int(row[0]), "content_type": row[1], "content": row[2]}
        for row in rows
    ]


def _row_to_message(row: sqlite3.Row | tuple) -> dict:
    return {
        "id": int(row[0]),
        "channel": row[1],
        "thread_id": row[2],
        "sender_id": row[3],
        "role": row[4],
        "timestamp": row[5],
    }


def get_conversation(
    conn: sqlite3.Connection,
    channel: str,
    thread_id: str,
) -> list[dict]:
    """Load all messages for channel+thread_id ordered by messages.id.

    Each dict has: id, channel, thread_id, sender_id, role, timestamp, content
    (a list of dicts with ordinal, content_type, content).
    """
    rows = conn.execute(
        """
        SELECT id, channel, thread_id, sender_id, role, timestamp
        FROM messages
        WHERE channel=? AND thread_id=?
        ORDER BY id ASC
        """,
        (channel, thread_id),
    ).fetchall()
    result: list[dict] = []
    for row in rows:
        message = _row_to_message(row)
        message["content"] = _load_content(conn, message["id"])
        result.append(message)
    return result


# --------------------------------------------------------------------------- #
# Email-specific functions
# --------------------------------------------------------------------------- #


def create_email_thread(
    conn: sqlite3.Connection,
    sender_id: str,
    created_ts: str | None = None,
) -> int:
    """Create a new email thread record. Returns email_threads.id."""
    ts = now_local_iso() if created_ts is None else created_ts
    cur = conn.execute(
        "INSERT INTO email_threads(sender_id, created_ts) VALUES(?, ?)",
        (sender_id, ts),
    )
    conn.commit()
    return int(cur.lastrowid)


def get_email_thread_sender(conn: sqlite3.Connection, thread_id: int) -> str | None:
    row = conn.execute(
        "SELECT sender_id FROM email_threads WHERE id=?",
        (thread_id,),
    ).fetchone()
    if row is None:
        return None
    return str(row[0])


def get_email_thread_by_message_id(
    conn: sqlite3.Connection,
    email_message_id: str,
) -> int | None:
    """Return email_threads.id for the thread containing this email_message_id."""
    row = conn.execute(
        """
        SELECT m.thread_id
        FROM email_message_meta meta
        JOIN messages m ON m.id = meta.message_id
        WHERE meta.email_message_id=? AND m.channel='email'
        LIMIT 1
        """,
        (email_message_id,),
    ).fetchone()
    if row is None:
        return None
    return int(row[0])


def resolve_email_thread_from_references(
    conn: sqlite3.Connection,
    *,
    sender_id: str,
    in_reply_to: str | None,
    references: list[str],
) -> int | None:
    """Return email_threads.id by looking up In-Reply-To and References headers."""
    refs_newest_first = list(reversed(list(references)))
    candidates: list[str] = []
    if in_reply_to:
        candidates.append(in_reply_to)
    candidates.extend(refs_newest_first)
    for message_id in candidates:
        thread_id = get_email_thread_by_message_id(conn, message_id)
        if thread_id is None:
            continue
        if get_email_thread_sender(conn, thread_id) == sender_id:
            return thread_id
    return None


def has_incoming_email_message_id(
    conn: sqlite3.Connection,
    email_message_id: str,
) -> bool:
    """Check if an email_message_id already exists in email_message_meta."""
    row = conn.execute(
        "SELECT 1 FROM email_message_meta WHERE email_message_id=? LIMIT 1",
        (email_message_id,),
    ).fetchone()
    return row is not None


def insert_email_message_meta(
    conn: sqlite3.Connection,
    *,
    message_id: int,
    email_message_id: str | None,
    in_reply_to: str | None = None,
    references_header: str | None = None,
    subject: str | None = None,
    process_state: str = COMPLETED,
    failure_details: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO email_message_meta(
            message_id,
            email_message_id,
            in_reply_to,
            references_header,
            subject,
            process_state,
            failure_details
        )
        VALUES(?, ?, ?, ?, ?, ?, ?)
        """,
        (
            message_id,
            email_message_id,
            in_reply_to,
            references_header,
            subject,
            process_state,
            failure_details,
        ),
    )
    conn.commit()


def update_email_message_state(
    conn: sqlite3.Connection,
    message_id: int,
    process_state: str,
    failure_details: str | None = None,
) -> None:
    conn.execute(
        "UPDATE email_message_meta SET process_state=?, failure_details=? WHERE message_id=?",
        (process_state, failure_details, message_id),
    )
    conn.commit()


def get_email_message_meta(conn: sqlite3.Connection, message_id: int) -> dict | None:
    """Return email_message_meta row as dict."""
    row = conn.execute(
        """
        SELECT message_id, email_message_id, in_reply_to, references_header,
               subject, process_state, failure_details
        FROM email_message_meta
        WHERE message_id=?
        """,
        (message_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "message_id": int(row[0]),
        "email_message_id": row[1],
        "in_reply_to": row[2],
        "references_header": row[3],
        "subject": row[4],
        "process_state": row[5],
        "failure_details": row[6],
    }


# --------------------------------------------------------------------------- #
# Email scheduler functions
# --------------------------------------------------------------------------- #


def list_email_threads_with_work(conn: sqlite3.Connection) -> list[int]:
    """Return distinct email_threads.id that have pending user messages."""
    rows = conn.execute(
        """
        SELECT DISTINCT m.thread_id
        FROM messages m
        JOIN email_message_meta meta ON meta.message_id = m.id
        WHERE m.channel='email' AND m.role='user' AND meta.process_state='pending'
        ORDER BY CAST(m.thread_id AS INTEGER) ASC
        """
    ).fetchall()
    return [int(row[0]) for row in rows]


def _get_next_pending_email_message(
    conn: sqlite3.Connection,
    thread_id_int: int,
) -> dict | None:
    row = conn.execute(
        """
        SELECT m.id, m.channel, m.thread_id, m.sender_id, m.role, m.timestamp,
               meta.email_message_id, meta.in_reply_to, meta.references_header, meta.subject
        FROM messages m
        JOIN email_message_meta meta ON meta.message_id = m.id
        WHERE m.channel='email' AND m.thread_id=? AND m.role='user'
              AND meta.process_state='pending'
        ORDER BY m.id ASC
        LIMIT 1
        """,
        (str(thread_id_int),),
    ).fetchone()
    if row is None:
        return None
    message = {
        "id": int(row[0]),
        "channel": row[1],
        "thread_id": row[2],
        "sender_id": row[3],
        "role": row[4],
        "timestamp": row[5],
        "email_message_id": row[6],
        "in_reply_to": row[7],
        "references_header": row[8],
        "subject": row[9],
    }
    message["content"] = _load_content(conn, message["id"])
    return message


def claim_next_pending_email_message(
    conn: sqlite3.Connection,
    thread_id_int: int,
) -> dict | None:
    """Atomically claim the oldest pending user message for this email thread.

    Returns a dict with message info + email meta, or None. The dict has:
    id, channel, thread_id, sender_id, role, timestamp, content (list),
    email_message_id, in_reply_to, references_header.
    """
    row = _get_next_pending_email_message(conn, thread_id_int)
    if row is None:
        return None
    updated = conn.execute(
        """
        UPDATE email_message_meta
        SET process_state='processing', failure_details=NULL
        WHERE message_id=? AND process_state='pending'
        """,
        (row["id"],),
    )
    conn.commit()
    if updated.rowcount != 1:
        return None
    return row


def reset_processing_email_messages(conn: sqlite3.Connection) -> None:
    """Reset any 'processing' email messages back to 'pending' (restart recovery)."""
    conn.execute(
        """
        UPDATE email_message_meta
        SET process_state=?, failure_details=NULL
        WHERE process_state=? AND message_id IN (
            SELECT id FROM messages WHERE channel='email' AND role='user'
        )
        """,
        (PENDING, PROCESSING),
    )
    conn.commit()


def requeue_failed_email_messages(conn: sqlite3.Connection) -> None:
    """Set all 'failed' email user messages back to 'pending'."""
    conn.execute(
        """
        UPDATE email_message_meta
        SET process_state=?
        WHERE process_state=? AND message_id IN (
            SELECT id FROM messages WHERE channel='email' AND role='user'
        )
        """,
        (PENDING, FAILED),
    )
    conn.commit()
