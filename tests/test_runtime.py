from __future__ import annotations

import asyncio
import sqlite3
import subprocess
import time

from backhead import db, mail
from backhead.main import RuntimeContext, create_podman_runner, schedule_conversation
from backhead.private_config import (
    AgentEndpointConfig,
    AppConfig,
    EmailAccountConfig,
    IMAPConfig,
    SMTPConfig,
)
from backhead.tools.cli_tool import create_cli_tool

SIMULATED_AGENT_DELAY = 0.1
MAX_CONCURRENT_PROCESSING_TIME = 0.18


def _config(maximum_concurrent_conversations: int = 2) -> AppConfig:
    return AppConfig(
        main_agent=AgentEndpointConfig(
            base_url="http://main",
            api_key="main-key",
            model="main-model",
        ),
        subagent=AgentEndpointConfig(
            base_url="http://sub",
            api_key="sub-key",
            model="sub-model",
        ),
        email_account=EmailAccountConfig(
            address="podhead@example.com",
            password="secret",
        ),
        imap=IMAPConfig(
            host="imap.example.com",
            port=993,
            username="imap-user",
            password="secret",
            inbox="INBOX",
            use_ssl=True,
        ),
        smtp=SMTPConfig(
            host="smtp.example.com",
            port=587,
            username="smtp-user",
            password="secret",
            use_tls=True,
        ),
        sender_whitelist=["alice@example.com", "bob@example.com"],
        mail_polling_interval_seconds=0.01,
        maximum_concurrent_conversations=maximum_concurrent_conversations,
        maximum_agent_depth=2,
        maximum_children_per_agent=4,
        podman_container_name="podhead-agent",
    )


def _conn(path: str | None = None):
    conn = sqlite3.connect(path or ":memory:", check_same_thread=False)
    db.init_db(conn)
    return conn


def _message_text(message_row: dict) -> str:
    return "".join(
        p["content"] for p in message_row.get("content", []) if p["content_type"] == "text"
    )


def _queue_message(
    conn,
    sender: str,
    message_id: str,
    body: str,
    *,
    timestamp: int,
    subject: str = "Hello",
    in_reply_to: str | None = None,
):
    return mail.store_incoming_email(
        conn=conn,
        incoming=mail.IncomingEmail(
            from_header=sender,
            subject=subject,
            content_parts=[mail.ContentPart(kind="text", text=body)],
            message_id=message_id,
            in_reply_to=in_reply_to,
            timestamp=timestamp,
        ),
        whitelist={"alice@example.com", "bob@example.com"},
    )


def test_async_processing_of_different_conversations(monkeypatch):
    conn = _conn()
    first = _queue_message(conn, "alice@example.com", "<a1@example.com>", "one", timestamp=1)
    second = _queue_message(conn, "bob@example.com", "<b1@example.com>", "two", timestamp=2)
    runtime = RuntimeContext(_config(maximum_concurrent_conversations=2), conn)
    monkeypatch.setattr(mail, "send_reply_smtp", lambda outgoing, smtp_config: None)

    def run_agent(history, message_row):
        time.sleep(SIMULATED_AGENT_DELAY)
        return f"reply:{_message_text(message_row)}"

    async def run_test():
        start = time.perf_counter()
        await schedule_conversation(runtime, first["email_thread_id"], run_agent, None)
        await schedule_conversation(runtime, second["email_thread_id"], run_agent, None)
        await asyncio.gather(*runtime.processing_tasks.values())
        return time.perf_counter() - start

    elapsed = asyncio.run(run_test())
    assert elapsed < MAX_CONCURRENT_PROCESSING_TIME


def test_fifo_processing_within_one_conversation(monkeypatch):
    conn = _conn()
    first = _queue_message(conn, "alice@example.com", "<a1@example.com>", "first", timestamp=1)
    _queue_message(
        conn,
        "alice@example.com",
        "<a2@example.com>",
        "second",
        timestamp=2,
        in_reply_to="<a1@example.com>",
    )
    runtime = RuntimeContext(_config(), conn)
    monkeypatch.setattr(mail, "send_reply_smtp", lambda outgoing, smtp_config: None)
    seen: list[str] = []

    def run_agent(history, message_row):
        seen.append(_message_text(message_row))
        time.sleep(0.05)
        return f"reply:{_message_text(message_row)}"

    async def run_test():
        await schedule_conversation(runtime, first["email_thread_id"], run_agent, None)
        await asyncio.gather(*runtime.processing_tasks.values())

    asyncio.run(run_test())
    assert seen == ["first", "second"]


def test_pending_work_survives_restart(monkeypatch, tmp_path):
    db_path = tmp_path / "agent.db"
    conn = _conn(str(db_path))
    queued = _queue_message(conn, "alice@example.com", "<restart@example.com>", "resume", timestamp=1)
    conn.close()

    restarted = _conn(str(db_path))
    runtime = RuntimeContext(_config(), restarted)
    monkeypatch.setattr(mail, "send_reply_smtp", lambda outgoing, smtp_config: None)

    def run_agent(history, message_row):
        return "done"

    async def run_test():
        await schedule_conversation(runtime, queued["email_thread_id"], run_agent, None)
        await asyncio.gather(*runtime.processing_tasks.values())

    asyncio.run(run_test())
    history = db.get_conversation(restarted, "email", str(queued["email_thread_id"]))
    user_states = [
        db.get_email_message_meta(restarted, row["id"])["process_state"]
        for row in history
        if row["role"] == "user"
    ]
    assert user_states == [db.COMPLETED]


def test_retry_after_processing_failure(monkeypatch):
    conn = _conn()
    queued = _queue_message(conn, "alice@example.com", "<retry@example.com>", "retry me", timestamp=1)
    runtime = RuntimeContext(_config(), conn)
    monkeypatch.setattr(mail, "send_reply_smtp", lambda outgoing, smtp_config: None)
    attempts = {"count": 0}

    def run_agent(history, message_row):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("temporary failure")
        return "done"

    async def run_once():
        await schedule_conversation(runtime, queued["email_thread_id"], run_agent, None)
        await asyncio.gather(*runtime.processing_tasks.values())

    asyncio.run(run_once())
    assert db.get_email_message_meta(conn, queued["message_id"])["process_state"] == db.FAILED

    db.requeue_failed_email_messages(conn)
    asyncio.run(run_once())
    assert db.get_email_message_meta(conn, queued["message_id"])["process_state"] == db.COMPLETED


def test_real_podman_command_routing_uses_exec_boundary(monkeypatch):
    calls = []

    def fake_run(command, capture_output, text, timeout, check):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="inside container", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = create_podman_runner("podhead-agent")
    cli_schema, cli_handler = create_cli_tool(runner)

    result = cli_handler({"command": "pwd"}, None)
    assert cli_schema["function"]["name"] == "run_cli"
    assert result == {"ok": True, "output": "inside container"}
    assert calls[0][:3] == ["podman", "exec", "podhead-agent"]


def test_processing_messages_are_requeued_after_restart():
    conn = _conn()
    queued = _queue_message(conn, "alice@example.com", "<processing@example.com>", "work", timestamp=1)
    db.update_email_message_state(conn, queued["message_id"], db.PROCESSING)
    db.reset_processing_email_messages(conn)
    assert db.get_email_message_meta(conn, queued["message_id"])["process_state"] == db.PENDING
