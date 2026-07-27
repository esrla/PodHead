from __future__ import annotations

import html
import sqlite3
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = PROJECT_ROOT / "state"
DB_PATH = STATE_DIR / "agent.db"
LOG_PATH = Path.home() / "podhead.log"

st.set_page_config(
    page_title="PodHead Admin",
    page_icon="🧠",
    layout="wide",
)


def connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(
        f"file:{DB_PATH}?mode=ro",
        uri=True,
        timeout=5,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def query(sql: str, parameters: tuple = ()) -> list[dict]:
    with connect_db() as conn:
        return [dict(row) for row in conn.execute(sql, parameters).fetchall()]


def scalar(sql: str, parameters: tuple = ()) -> int:
    rows = query(sql, parameters)
    return int(next(iter(rows[0].values()))) if rows else 0


def table_exists(table_name: str) -> bool:
    if not DB_PATH.exists():
        return False

    return bool(
        query(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table' AND name = ?
            """,
            (table_name,),
        )
    )


def render_message_content(message_id: int) -> None:
    parts = query(
        """
        SELECT content_type, content
        FROM message_content
        WHERE message_id = ?
        ORDER BY ordinal
        """,
        (message_id,),
    )

    if not parts:
        st.caption("No stored content")
        return

    for part in parts:
        content_type = part["content_type"]
        content = part["content"]

        if content_type == "text":
            st.markdown(content)
            continue

        if content_type == "image":
            image_path = (STATE_DIR / content).resolve()
            try:
                image_path.relative_to(STATE_DIR.resolve())
            except ValueError:
                st.warning(f"Invalid image path: {content}")
                continue

            if image_path.is_file():
                st.image(str(image_path), caption=content)
            else:
                st.warning(f"Image not found: {content}")
            continue

        st.code(content, language=None)


def status_page() -> None:
    st.header("Status")

    if not DB_PATH.exists():
        st.error(f"Database not found: {DB_PATH}")
        return

    message_count = scalar("SELECT COUNT(*) FROM messages")
    conversation_count = scalar(
        """
        SELECT COUNT(*)
        FROM (
            SELECT channel, thread_id
            FROM messages
            GROUP BY channel, thread_id
        )
        """
    )
    sender_count = scalar("SELECT COUNT(DISTINCT sender_id) FROM messages")
    embedding_count = (
        scalar("SELECT COUNT(*) FROM message_embeddings")
        if table_exists("message_embeddings")
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Messages", message_count)
    col2.metric("Conversations", conversation_count)
    col3.metric("Senders", sender_count)
    col4.metric("Embeddings", embedding_count)

    st.subheader("Latest activity")

    rows = query(
        """
        SELECT
            m.id,
            m.timestamp,
            m.channel,
            m.thread_id,
            m.sender_id,
            m.role,
            COALESCE(
                GROUP_CONCAT(
                    CASE
                        WHEN c.content_type = 'text' THEN c.content
                        ELSE '[' || c.content_type || ']'
                    END,
                    ' '
                ),
                ''
            ) AS content
        FROM messages AS m
        LEFT JOIN message_content AS c ON c.message_id = m.id
        GROUP BY m.id
        ORDER BY m.id DESC
        LIMIT 25
        """
    )

    st.dataframe(
        rows,
        width="stretch",
        hide_index=True,
        column_config={
            "content": st.column_config.TextColumn("Content", width="large"),
            "thread_id": st.column_config.TextColumn("Thread", width="medium"),
        },
    )


def conversations_page() -> None:
    st.header("Conversations")

    if not DB_PATH.exists():
        st.error(f"Database not found: {DB_PATH}")
        return

    conversations = query(
        """
        SELECT
            channel,
            thread_id,
            sender_id,
            COUNT(*) AS messages,
            MIN(timestamp) AS started_at,
            MAX(timestamp) AS latest_at
        FROM messages
        GROUP BY channel, thread_id, sender_id
        ORDER BY MAX(id) DESC
        """
    )

    if not conversations:
        st.info("No conversations found.")
        return

    search = st.text_input(
        "Filter",
        placeholder="Channel, thread or sender",
    ).strip().lower()

    filtered = [
        conversation
        for conversation in conversations
        if not search
        or search
        in " ".join(
            str(conversation[field]).lower()
            for field in ("channel", "thread_id", "sender_id")
        )
    ]

    if not filtered:
        st.info("No conversations match the filter.")
        return

    labels = [
        (
            f'{conversation["latest_at"]} · '
            f'{conversation["channel"]} · '
            f'{conversation["sender_id"]} · '
            f'{conversation["messages"]} messages · '
            f'{conversation["thread_id"]}'
        )
        for conversation in filtered
    ]

    selected_label = st.selectbox("Conversation", labels)
    selected = filtered[labels.index(selected_label)]

    metadata_col1, metadata_col2, metadata_col3 = st.columns(3)
    metadata_col1.metric("Messages", selected["messages"])
    metadata_col2.text(f'Channel: {selected["channel"]}')
    metadata_col3.text(f'Sender: {selected["sender_id"]}')

    if table_exists("conversation_compactions"):
        compactions = query(
            """
            SELECT
                upto_message_id,
                summary,
                summary_format,
                compacted_at
            FROM conversation_compactions
            WHERE channel = ? AND thread_id = ?
            """,
            (selected["channel"], selected["thread_id"]),
        )

        if compactions:
            with st.expander("Compaction summary"):
                compaction = compactions[0]
                st.caption(
                    f'Compacted through message {compaction["upto_message_id"]} '
                    f'· {compaction["compacted_at"]} '
                    f'· {compaction["summary_format"]}'
                )
                st.markdown(compaction["summary"])

    messages = query(
        """
        SELECT id, role, sender_id, timestamp
        FROM messages
        WHERE channel = ? AND thread_id = ?
        ORDER BY id
        """,
        (selected["channel"], selected["thread_id"]),
    )

    st.subheader("Messages")

    for message in messages:
        role = str(message["role"]).lower()
        chat_role = "assistant" if role == "assistant" else "user"

        with st.chat_message(chat_role):
            st.caption(
                f'#{message["id"]} · {message["role"]} · '
                f'{message["sender_id"]} · {message["timestamp"]}'
            )
            render_message_content(message["id"])


def embeddings_page() -> None:
    st.header("Embeddings")

    if not DB_PATH.exists():
        st.error(f"Database not found: {DB_PATH}")
        return

    if not table_exists("message_embeddings"):
        st.info("The embedding table does not exist.")
        return

    total_messages = scalar("SELECT COUNT(*) FROM messages")
    embedded_messages = scalar("SELECT COUNT(*) FROM message_embeddings")
    coverage = embedded_messages / total_messages * 100 if total_messages else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Messages", total_messages)
    col2.metric("Embedded", embedded_messages)
    col3.metric("Coverage", f"{coverage:.1f}%")

    models = query(
        """
        SELECT embedding_model, COUNT(*) AS messages
        FROM message_embeddings
        GROUP BY embedding_model
        ORDER BY messages DESC
        """
    )

    if models:
        st.subheader("Models")
        st.dataframe(models, width="stretch", hide_index=True)

    st.subheader("Latest embeddings")

    rows = query(
        """
        SELECT
            e.message_id,
            e.embedding_model,
            e.embedded_at,
            m.channel,
            m.thread_id,
            m.sender_id,
            m.role,
            e.searchable_text
        FROM message_embeddings AS e
        JOIN messages AS m ON m.id = e.message_id
        ORDER BY e.message_id DESC
        LIMIT 100
        """
    )

    st.dataframe(
        rows,
        width="stretch",
        hide_index=True,
        column_config={
            "searchable_text": st.column_config.TextColumn(
                "Searchable text",
                width="large",
            )
        },
    )


def logs_page() -> None:
    st.header("Logs")
    st.caption(str(LOG_PATH))

    if not LOG_PATH.is_file():
        st.warning("Log file not found.")
        return

    line_count = st.number_input(
        "Lines",
        min_value=50,
        max_value=5000,
        value=500,
        step=50,
    )
    search = st.text_input("Filter log").strip().lower()

    lines = LOG_PATH.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()[-int(line_count):]

    if search:
        lines = [line for line in lines if search in line.lower()]

    st.code("\n".join(lines) or "No matching log lines.", language="text")


st.title("PodHead Admin")

page = st.sidebar.radio(
    "View",
    ("Status", "Conversations", "Embeddings", "Logs"),
)

st.sidebar.caption(f"Database: {DB_PATH}")
st.sidebar.caption(f"Log: {LOG_PATH}")

try:
    if page == "Status":
        status_page()
    elif page == "Conversations":
        conversations_page()
    elif page == "Embeddings":
        embeddings_page()
    else:
        logs_page()
except sqlite3.Error as error:
    st.error(f"Database error: {html.escape(str(error))}")
