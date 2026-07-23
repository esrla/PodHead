"""PodHead backend runtime loop."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import subprocess
import threading
import traceback
from typing import Any, Callable

from backhead import db, mail
from backhead import media as media_mod
from backhead.agent_loop import Agent, DEFAULT_SYSTEM_PROMPT, messages_to_openai_messages
from backhead.bootstrap import STATE_DIR
from backhead.bootstrap import ensure_runtime, open_db_connection, resolve_workspace_host_path
from backhead.embeddings import create_openai_embed_fn
from backhead.llm import create_openai_client
from backhead.private_config import CONFIG, AppConfig
from backhead.skills import generate_skill_header
from backhead.tools.cli_tool import create_cli_tool
from backhead.tools.embed_tool import create_embed_tool
from backhead.tools.spawn_subagent import create_spawn_subagent_tool

MEDIA_ROOT = STATE_DIR


class ContainerExecutionError(RuntimeError):
    error_type = "container_execution_error"


@dataclass(frozen=True)
class QueuedEmailJob:
    message_id: int
    transport: mail.EmailTransportData


class RuntimeContext:
    def __init__(self, config: AppConfig, conn: sqlite3.Connection) -> None:
        self.config = config
        self.conn = conn
        self.db_lock = threading.Lock()
        self.processing_tasks: dict[str, asyncio.Task] = {}
        self.processing_tasks_lock = asyncio.Lock()
        self.queue_lock = asyncio.Lock()
        self.thread_queues: dict[str, deque[QueuedEmailJob]] = defaultdict(deque)
        self.semaphore = asyncio.Semaphore(config.maximum_concurrent_conversations)


def _db_call(runtime: RuntimeContext, func: Callable[..., Any], *args, **kwargs):
    with runtime.db_lock:
        return func(runtime.conn, *args, **kwargs)


def _current_message_prompt(current_message: dict, media_root: Path | None):
    """Build an OpenAI prompt (str or content list) from a stored message."""
    oai_parts: list[dict] = []
    for part in current_message.get("content", []):
        ct = part["content_type"]
        val = part["content"]
        if ct == "text":
            oai_parts.append({"type": "text", "text": val})
        elif ct == "image":
            if media_root:
                b64 = media_mod.load_image_as_base64(val, media_root)
                if b64:
                    mime = media_mod.get_image_mime_type(val)
                    oai_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        }
                    )
                else:
                    oai_parts.append({"type": "text", "text": "[image not found]"})
            else:
                oai_parts.append({"type": "text", "text": f"[image: {val}]"})
    if not oai_parts:
        return ""
    if all(p["type"] == "text" for p in oai_parts):
        return "".join(p["text"] for p in oai_parts)
    return oai_parts


def build_email_agent_runner(
    *,
    openai_client,
    model: str,
    system_prompt: str,
    tools: list,
    tool_handlers: dict,
    container_runner,
    max_depth: int,
    max_children: int,
    media_root: Path | None = None,
    workspace_path: Path | None = None,
    skill_header_provider: Any = None,
):
    def run_agent(history: list[dict], current_message: dict) -> str:
        prior = [row for row in history if row["id"] != current_message["id"]]
        prior_messages = messages_to_openai_messages(prior, media_root=media_root)

        agent = Agent(
            openai_client=openai_client,
            model=model,
            system_prompt=system_prompt,
            workspace_path=workspace_path,
            skill_header_provider=skill_header_provider,
            conversation_history=prior_messages,
            tools=tools,
            tool_handlers=tool_handlers,
            container_runner=container_runner,
            depth=0,
            max_depth=max_depth,
            max_children=max_children,
        )
        prompt = _current_message_prompt(current_message, media_root)
        return agent.run(prompt)

    return run_agent


def create_podman_runner(container_name: str, timeout_seconds: int = 300):
    def run_in_container(command: str) -> str:
        quoted = f"cd /workspace && {command}"
        try:
            completed = subprocess.run(
                ["podman", "exec", container_name, "/bin/sh", "-lc", quoted],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"Command timed out after {timeout_seconds} seconds.") from exc
        except Exception as exc:  # noqa: BLE001
            raise ContainerExecutionError(str(exc)) from exc

        output = (completed.stdout or "") + (completed.stderr or "")
        if completed.returncode != 0:
            raise ContainerExecutionError(output.strip() or f"Command exited with {completed.returncode}.")
        return output.strip()

    return run_in_container


def create_tooling(*, config: AppConfig, container_runner):
    main_client = create_openai_client(config.main_agent.base_url, config.main_agent.api_key)
    sub_client = create_openai_client(config.subagent.base_url, config.subagent.api_key)
    embedding_client = create_openai_client(config.main_agent.base_url, config.main_agent.api_key)
    embed_fn = create_openai_embed_fn(embedding_client, config.embedding_model)

    cli_schema, cli_handler = create_cli_tool(container_runner)
    embed_schema, embed_handler = create_embed_tool(embed_fn)

    def skill_header_provider(prompt_text: str, workspace_path: Path) -> str | None:
        try:
            return generate_skill_header(
                prompt_text,
                workspace_path,
                min_similarity=config.skill_similarity_threshold,
                embed_fn=embed_fn,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Skill-header generation failed: {exc}")
            return None

    subagent_tools: list[dict] = [cli_schema, embed_schema]
    subagent_handlers: dict[str, Any] = {"run_cli": cli_handler, "embed_text": embed_handler}
    spawn_schema, spawn_handler = create_spawn_subagent_tool(
        openai_client=sub_client,
        model=config.subagent.model,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        workspace_path=resolve_workspace_host_path(config),
        skill_header_provider=skill_header_provider,
        tools=subagent_tools,
        tool_handlers=subagent_handlers,
        container_runner=container_runner,
        max_depth=config.maximum_agent_depth,
        max_children=config.maximum_children_per_agent,
    )
    subagent_tools.append(spawn_schema)
    subagent_handlers["spawn_subagent"] = spawn_handler

    main_tools = [cli_schema, spawn_schema, embed_schema]
    main_handlers = {"run_cli": cli_handler, "spawn_subagent": spawn_handler, "embed_text": embed_handler}

    return {
        "main_client": main_client,
        "sub_client": sub_client,
        "main_tools": main_tools,
        "main_handlers": main_handlers,
        "sub_tools": subagent_tools,
        "sub_handlers": subagent_handlers,
        "skill_header_provider": skill_header_provider,
    }


async def _pop_next_job(runtime: RuntimeContext, thread_id: str) -> QueuedEmailJob | None:
    async with runtime.queue_lock:
        queue = runtime.thread_queues.get(thread_id)
        if not queue:
            return None
        return queue.popleft() if queue else None


async def _process_conversation(runtime: RuntimeContext, thread_id: str, run_agent) -> None:
    async with runtime.semaphore:
        while True:
            job = await _pop_next_job(runtime, thread_id)
            if job is None:
                return

            history = await asyncio.to_thread(_db_call, runtime, db.get_conversation, "email", thread_id)
            current_message = next((m for m in history if m["id"] == job.message_id), None)
            if current_message is None:
                continue

            try:
                reply_text = await asyncio.to_thread(run_agent, history, current_message)
                outgoing, _ = mail.build_reply_email(
                    from_address=runtime.config.email_account.address,
                    to=job.transport.sender_id,
                    subject=job.transport.subject,
                    body=reply_text,
                    thread_id=thread_id,
                    incoming_message_id=job.transport.incoming_message_id,
                    references_header=job.transport.references,
                )
                await asyncio.to_thread(mail.send_reply_smtp, outgoing, runtime.config.smtp)
                ts = db.now_local_iso()
                await asyncio.to_thread(
                    _db_call,
                    runtime,
                    db.insert_message_with_content,
                    channel="email",
                    thread_id=thread_id,
                    sender_id=job.transport.sender_id,
                    role="assistant",
                    timestamp=ts,
                    content_parts=[("text", reply_text)],
                )
            except Exception:  # noqa: BLE001
                traceback_text = traceback.format_exc()
                print(f"Failed to process message {job.message_id} in thread {thread_id}:\n{traceback_text}", end="")
                await _send_request_error_response(runtime, job, thread_id, traceback_text)


async def _send_request_error_response(
    runtime: RuntimeContext,
    job: QueuedEmailJob,
    thread_id: str,
    traceback_text: str,
) -> None:
    try:
        outgoing, _ = mail.build_reply_email(
            from_address=runtime.config.email_account.address,
            to=job.transport.sender_id,
            subject=job.transport.subject,
            body=traceback_text,
            thread_id=thread_id,
            incoming_message_id=job.transport.incoming_message_id,
            references_header=job.transport.references,
        )
        await asyncio.to_thread(mail.send_reply_smtp, outgoing, runtime.config.smtp)
    except Exception as exc:  # noqa: BLE001
        print(
            f"Failed to send email request error response for message {job.message_id} "
            f"in thread {thread_id}: {exc!r}"
        )
        return

    try:
        ts = db.now_local_iso()
        await asyncio.to_thread(
            _db_call,
            runtime,
            db.insert_message_with_content,
            channel="email",
            thread_id=thread_id,
            sender_id=job.transport.sender_id,
            role="assistant",
            timestamp=ts,
            content_parts=[("text", traceback_text)],
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"Failed to persist request error response for message {job.message_id} "
            f"in thread {thread_id}: {exc!r}"
        )


async def schedule_conversation(runtime: RuntimeContext, thread_id: str, run_agent) -> None:
    async with runtime.processing_tasks_lock:
        existing = runtime.processing_tasks.get(thread_id)
        if existing and not existing.done():
            return
        task = asyncio.create_task(_process_conversation(runtime, thread_id, run_agent))
        runtime.processing_tasks[thread_id] = task

        def _cleanup(_task, thread_id=thread_id):
            runtime.processing_tasks.pop(thread_id, None)

        task.add_done_callback(_cleanup)


def _poll_inbox(runtime: RuntimeContext) -> list[dict]:
    return mail.poll_inbox(
        conn=runtime.conn,
        whitelist=runtime.config.sender_whitelist,
        imap_config=runtime.config.imap,
        spam_mailbox=runtime.config.spam_mailbox,
        media_root=MEDIA_ROOT,
        db_lock=runtime.db_lock,
    )


async def poll_and_schedule(runtime: RuntimeContext, run_agent) -> None:
    results = await asyncio.to_thread(_poll_inbox, runtime)
    queued_by_thread: dict[str, list[QueuedEmailJob]] = defaultdict(list)
    for result in results:
        if result.get("status") != "queued":
            continue
        transport = result.get("transport")
        thread_id = result.get("thread_id")
        message_id = result.get("message_id")
        if not isinstance(transport, mail.EmailTransportData):
            continue
        if not isinstance(thread_id, str) or not isinstance(message_id, int):
            continue
        queued_by_thread[thread_id].append(QueuedEmailJob(message_id=message_id, transport=transport))

    async with runtime.queue_lock:
        for thread_id, jobs in queued_by_thread.items():
            runtime.thread_queues[thread_id].extend(jobs)

    for thread_id in sorted(queued_by_thread):
        await schedule_conversation(runtime, thread_id, run_agent)


async def run_backend(config: AppConfig = CONFIG) -> None:
    conn = open_db_connection()
    runtime = RuntimeContext(config, conn)
    container_runner = create_podman_runner(config.podman_container_name)
    tooling = create_tooling(config=config, container_runner=container_runner)
    run_agent = build_email_agent_runner(
        openai_client=tooling["main_client"],
        model=config.main_agent.model,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        tools=tooling["main_tools"],
        tool_handlers=tooling["main_handlers"],
        container_runner=container_runner,
        max_depth=config.maximum_agent_depth,
        max_children=config.maximum_children_per_agent,
        media_root=MEDIA_ROOT,
        workspace_path=resolve_workspace_host_path(config),
        skill_header_provider=tooling["skill_header_provider"],
    )

    while True:
        await poll_and_schedule(runtime, run_agent)
        await asyncio.sleep(config.mail_polling_interval_seconds)


def main() -> None:
    ensure_runtime(CONFIG)
    asyncio.run(run_backend(CONFIG))


if __name__ == "__main__":
    main()
