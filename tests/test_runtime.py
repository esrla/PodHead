from __future__ import annotations

import asyncio
import sqlite3
import subprocess
import time

from backhead import db, mail
from backhead.main import QueuedEmailJob, RuntimeContext, create_podman_runner, schedule_conversation
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
        main_agent=AgentEndpointConfig(base_url="http://main", api_key="main-key", model="main-model"),
        subagent=AgentEndpointConfig(base_url="http://sub", api_key="sub-key", model="sub-model"),
        email_account=EmailAccountConfig(address="podhead@example.com", **{"password": "mail-password"}),
        imap=IMAPConfig(
            host="imap.example.com",
            port=993,
            username="imap-user",
            **{"password": "imap-password"},
            inbox="INBOX",
            use_ssl=True,
        ),
        smtp=SMTPConfig(
            host="smtp.example.com",
            port=587,
            username="smtp-user",
            **{"password": "smtp-password"},
            use_tls=True,
        ),
        sender_whitelist=["alice@example.com", "bob@example.com"],
        mail_polling_interval_seconds=0.01,
        maximum_concurrent_conversations=maximum_concurrent_conversations,
        maximum_agent_depth=2,
        maximum_children_per_agent=4,
        embedding_model="embed-model",
        skill_similarity_threshold=0.35,
        podman_container_name="podhead-agent",
        workspace_path="head_pod",
        spam_mailbox="Junk",
    )


def _conn(path: str | None = None):
    conn = sqlite3.connect(path or ":memory:", check_same_thread=False)
    db.init_db(conn)
    return conn


def _message_text(message_row: dict) -> str:
    return "".join(p["content"] for p in message_row.get("content", []) if p["content_type"] == "text")


def _queue_message(conn, sender: str, message_id: str, body: str, *, timestamp: int):
    return mail.store_incoming_email(
        conn=conn,
        incoming=mail.IncomingEmail(
            from_header=sender,
            subject="Hello",
            content_parts=[mail.ContentPart(kind="text", text=body)],
            message_id=message_id,
            timestamp=timestamp,
        ),
        whitelist={"alice@example.com", "bob@example.com"},
    )


def _enqueue(runtime: RuntimeContext, queued: dict):
    runtime.thread_queues[queued["thread_id"]].append(
        QueuedEmailJob(message_id=queued["message_id"], transport=queued["transport"])
    )


def test_async_processing_of_different_conversations(monkeypatch):
    conn = _conn()
    first = _queue_message(conn, "alice@example.com", "<a1@example.com>", "one", timestamp=1)
    second = _queue_message(conn, "bob@example.com", "<b1@example.com>", "two", timestamp=2)
    runtime = RuntimeContext(_config(maximum_concurrent_conversations=2), conn)
    monkeypatch.setattr(mail, "send_reply_smtp", lambda outgoing, smtp_config: None)
    _enqueue(runtime, first)
    _enqueue(runtime, second)

    def run_agent(history, message_row):
        time.sleep(SIMULATED_AGENT_DELAY)
        return f"reply:{_message_text(message_row)}"

    async def run_test():
        start = time.perf_counter()
        await schedule_conversation(runtime, first["thread_id"], run_agent)
        await schedule_conversation(runtime, second["thread_id"], run_agent)
        await asyncio.gather(*runtime.processing_tasks.values())
        return time.perf_counter() - start

    elapsed = asyncio.run(run_test())
    assert elapsed < MAX_CONCURRENT_PROCESSING_TIME


def test_fifo_processing_within_one_conversation(monkeypatch):
    conn = _conn()
    first = _queue_message(conn, "alice@example.com", "<a1@example.com>", "first", timestamp=1)
    second = mail.store_incoming_email(
        conn=conn,
        incoming=mail.IncomingEmail(
            from_header="alice@example.com",
            subject="Hello",
            content_parts=[mail.ContentPart(kind="text", text="second")],
            message_id="<a2@example.com>",
            in_reply_to=f"<podhead.{first['thread_id']}.token@example.com>",
            timestamp=2,
        ),
        whitelist={"alice@example.com", "bob@example.com"},
    )
    runtime = RuntimeContext(_config(), conn)
    monkeypatch.setattr(mail, "send_reply_smtp", lambda outgoing, smtp_config: None)
    _enqueue(runtime, first)
    _enqueue(runtime, second)
    seen: list[str] = []

    def run_agent(history, message_row):
        seen.append(_message_text(message_row))
        time.sleep(0.05)
        return f"reply:{_message_text(message_row)}"

    async def run_test():
        await schedule_conversation(runtime, first["thread_id"], run_agent)
        await asyncio.gather(*runtime.processing_tasks.values())

    asyncio.run(run_test())
    assert seen == ["first", "second"]


def test_failure_is_logged_without_persistent_retry(monkeypatch):
    conn = _conn()
    queued = _queue_message(conn, "alice@example.com", "<retry@example.com>", "retry me", timestamp=1)
    runtime = RuntimeContext(_config(), conn)
    monkeypatch.setattr(mail, "send_reply_smtp", lambda outgoing, smtp_config: None)
    _enqueue(runtime, queued)

    def run_agent(history, message_row):
        raise RuntimeError("temporary failure")

    async def run_once():
        await schedule_conversation(runtime, queued["thread_id"], run_agent)
        await asyncio.gather(*runtime.processing_tasks.values())

    asyncio.run(run_once())
    history = db.get_conversation(conn, "email", queued["thread_id"])
    assert [row["role"] for row in history] == ["user"]


def test_assistant_sender_id_matches_conversation_owner(monkeypatch):
    conn = _conn()
    queued = _queue_message(conn, "alice@example.com", "<a1@example.com>", "first", timestamp=1)
    runtime = RuntimeContext(_config(), conn)
    monkeypatch.setattr(mail, "send_reply_smtp", lambda outgoing, smtp_config: None)
    _enqueue(runtime, queued)

    async def run_test():
        await schedule_conversation(runtime, queued["thread_id"], lambda history, message_row: "done")
        await asyncio.gather(*runtime.processing_tasks.values())

    asyncio.run(run_test())
    history = db.get_conversation(conn, "email", queued["thread_id"])
    assistant = next(row for row in history if row["role"] == "assistant")
    assert assistant["sender_id"] == "alice@example.com"


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
