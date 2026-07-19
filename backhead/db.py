"""SQLite schema and access helpers for the generic conversation log."""

from __future__ import annotations

import sqlite3
from datetime import datetime


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
    conn.commit()


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
    """Atomically insert a message and all its content rows. Returns messages.id."""
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
    """Load all messages for channel+thread_id ordered by messages.id."""
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
