"""SQLite schema and access helpers for the generic conversation log."""

from __future__ import annotations

from collections.abc import Callable
import sqlite3
from datetime import datetime


MESSAGE_SEARCHABLE_TEXT_FORMAT_VERSION = "chat-history-searchable-text-v1"
COMPACTION_SUMMARY_FORMAT_VERSION = "conversation-compaction-v1"


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
        CREATE TABLE IF NOT EXISTS message_embeddings (
            message_id INTEGER PRIMARY KEY,
            embedding_model TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            searchable_text TEXT NOT NULL,
            embedding BLOB NOT NULL,
            embedded_at TEXT NOT NULL,
            FOREIGN KEY(message_id) REFERENCES messages(id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_compactions (
            channel TEXT NOT NULL,
            thread_id TEXT NOT NULL,
            sender_id TEXT NOT NULL,
            upto_message_id INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            summary TEXT NOT NULL,
            summary_format TEXT NOT NULL,
            compacted_at TEXT NOT NULL,
            PRIMARY KEY(channel, thread_id)
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
        CREATE INDEX IF NOT EXISTS idx_messages_sender_id
        ON messages(sender_id, id)
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
        CREATE INDEX IF NOT EXISTS idx_compactions_sender_id
        ON conversation_compactions(sender_id, compacted_at)
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


def load_message_parts(conn: sqlite3.Connection, message_id: int) -> list[dict]:
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
        message["content"] = load_message_parts(conn, message["id"])
        result.append(message)
    return result


def find_oldest_message_needing_embedding(
    conn: sqlite3.Connection,
    *,
    sender_id: str,
    embedding_model: str,
    searchable_text_builder: Callable[[dict], str],
    content_hash_builder: Callable[[dict, str, str], str],
) -> dict | None:
    rows = conn.execute(
        """
        SELECT id, channel, thread_id, sender_id, role, timestamp
        FROM messages
        WHERE sender_id=?
        ORDER BY id ASC
        """,
        (sender_id,),
    ).fetchall()
    for row in rows:
        message = _row_to_message(row)
        message["content"] = load_message_parts(conn, message["id"])
        searchable_text = searchable_text_builder(message)
        content_hash = content_hash_builder(message, searchable_text, embedding_model)
        existing = conn.execute(
            """
            SELECT embedding_model, content_hash, searchable_text
            FROM message_embeddings
            WHERE message_id=?
            """,
            (message["id"],),
        ).fetchone()
        if existing is None:
            return {**message, "searchable_text": searchable_text, "content_hash": content_hash}
        if existing[0] != embedding_model:
            return {**message, "searchable_text": searchable_text, "content_hash": content_hash}
        if existing[1] != content_hash:
            return {**message, "searchable_text": searchable_text, "content_hash": content_hash}
        if existing[2] != searchable_text:
            return {**message, "searchable_text": searchable_text, "content_hash": content_hash}
    return None


def store_or_replace_message_embedding(
    conn: sqlite3.Connection,
    *,
    message_id: int,
    embedding_model: str,
    content_hash: str,
    searchable_text: str,
    embedding: bytes,
    embedded_at: str,
) -> None:
    try:
        conn.execute(
            """
            INSERT INTO message_embeddings(
                message_id,
                embedding_model,
                content_hash,
                searchable_text,
                embedding,
                embedded_at
            )
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(message_id) DO UPDATE SET
                embedding_model=excluded.embedding_model,
                content_hash=excluded.content_hash,
                searchable_text=excluded.searchable_text,
                embedding=excluded.embedding,
                embedded_at=excluded.embedded_at
            """,
            (message_id, embedding_model, content_hash, searchable_text, sqlite3.Binary(embedding), embedded_at),
        )
    except Exception:
        conn.rollback()
        raise
    conn.commit()


def get_message_embeddings_for_sender(
    conn: sqlite3.Connection,
    *,
    sender_id: str,
    exclude_thread_id: str | None = None,
) -> list[dict]:
    query = """
        SELECT
            m.id,
            m.channel,
            m.thread_id,
            m.sender_id,
            m.role,
            m.timestamp,
            e.embedding_model,
            e.content_hash,
            e.searchable_text,
            e.embedding,
            e.embedded_at
        FROM messages AS m
        JOIN message_embeddings AS e ON e.message_id = m.id
        WHERE m.sender_id=?
    """
    params: list[object] = [sender_id]
    if exclude_thread_id is not None:
        query += " AND m.thread_id<>?"
        params.append(exclude_thread_id)
    query += " ORDER BY m.id ASC"
    rows = conn.execute(query, tuple(params)).fetchall()
    return [
        {
            "message_id": int(row[0]),
            "channel": row[1],
            "thread_id": row[2],
            "sender_id": row[3],
            "role": row[4],
            "timestamp": row[5],
            "embedding_model": row[6],
            "content_hash": row[7],
            "searchable_text": row[8],
            "embedding": row[9],
            "embedded_at": row[10],
        }
        for row in rows
    ]


def get_conversation_compaction(
    conn: sqlite3.Connection,
    *,
    channel: str,
    thread_id: str,
) -> dict | None:
    row = conn.execute(
        """
        SELECT channel, thread_id, sender_id, upto_message_id, content_hash, summary, summary_format, compacted_at
        FROM conversation_compactions
        WHERE channel=? AND thread_id=?
        """,
        (channel, thread_id),
    ).fetchone()
    if row is None:
        return None
    return {
        "channel": row[0],
        "thread_id": row[1],
        "sender_id": row[2],
        "upto_message_id": int(row[3]),
        "content_hash": row[4],
        "summary": row[5],
        "summary_format": row[6],
        "compacted_at": row[7],
    }


def store_or_replace_conversation_compaction(
    conn: sqlite3.Connection,
    *,
    channel: str,
    thread_id: str,
    sender_id: str,
    upto_message_id: int,
    content_hash: str,
    summary: str,
    summary_format: str,
    compacted_at: str,
) -> None:
    try:
        conn.execute(
            """
            INSERT INTO conversation_compactions(
                channel,
                thread_id,
                sender_id,
                upto_message_id,
                content_hash,
                summary,
                summary_format,
                compacted_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(channel, thread_id) DO UPDATE SET
                sender_id=excluded.sender_id,
                upto_message_id=excluded.upto_message_id,
                content_hash=excluded.content_hash,
                summary=excluded.summary,
                summary_format=excluded.summary_format,
                compacted_at=excluded.compacted_at
            """,
            (
                channel,
                thread_id,
                sender_id,
                upto_message_id,
                content_hash,
                summary,
                summary_format,
                compacted_at,
            ),
        )
    except Exception:
        conn.rollback()
        raise
    conn.commit()
