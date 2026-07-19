"""PodHead backend entrypoint."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sqlite3
import subprocess
import threading
from typing import Any, Callable

from backhead import db, mail
from backhead import media as media_mod
from backhead.agent_loop import Agent, DEFAULT_SYSTEM_PROMPT, messages_to_openai_messages
from backhead.llm import create_openai_client, test_openai_endpoint
from backhead.private_config import CONFIG, AppConfig
from backhead.tools.cli_tool import create_cli_tool
from backhead.tools.spawn_subagent import create_spawn_subagent_tool

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = REPO_ROOT / "state"
DB_PATH = STATE_DIR / "agent.db"
MEDIA_ROOT = STATE_DIR
CONTAINER_IMAGE_NAME = "podhead-agent-image"
WORKSPACE_HOST_PATH = (REPO_ROOT / "head_pod" / "workspace").resolve()
PRIVATE_CONFIG_PATH = Path(__file__).resolve().parent / "private_config.py"
BACKEND_PATH = (REPO_ROOT / "backhead").resolve()


class ContainerExecutionError(RuntimeError):
    error_type = "container_execution_error"


class PodmanVerificationError(RuntimeError):
    pass


class RuntimeContext:
    def __init__(self, config: AppConfig, conn: sqlite3.Connection) -> None:
        self.config = config
        self.conn = conn
        self.db_lock = threading.Lock()
        self.processing_tasks: dict[int, asyncio.Task] = {}
        self.processing_tasks_lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(config.maximum_concurrent_conversations)



def _db_call(runtime: RuntimeContext, func: Callable[..., Any], *args, **kwargs):
    with runtime.db_lock:
        return func(runtime.conn, *args, **kwargs)



def ensure_state_dir() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)



def open_db_connection(path: Path = DB_PATH) -> sqlite3.Connection:
    ensure_state_dir()
    conn = sqlite3.connect(path, check_same_thread=False)
    db.init_db(conn)
    return conn



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
):
    def run_agent(history: list[dict], current_message: dict) -> str:
        prior = [row for row in history if row["id"] != current_message["id"]]
        prior_messages = messages_to_openai_messages(prior, media_root=media_root)
        agent = Agent(
            openai_client=openai_client,
            model=model,
            system_prompt=system_prompt,
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



def _podman_inspect(name: str) -> dict:
    completed = subprocess.run(
        ["podman", "inspect", name],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise PodmanVerificationError(completed.stderr.strip() or f"Container {name!r} does not exist.")
    import json

    items = json.loads(completed.stdout)
    if not items:
        raise PodmanVerificationError(f"Container {name!r} does not exist.")
    return items[0]



def verify_container_environment(config: AppConfig) -> None:
    inspect = _podman_inspect(config.podman_container_name)
    state = inspect.get("State") or {}
    if not state.get("Running"):
        raise PodmanVerificationError(f"Container {config.podman_container_name!r} is not running.")

    mounts = inspect.get("Mounts") or []
    workspace_mount = None
    for mount in mounts:
        destination = Path(mount.get("Destination", ""))
        source = Path(mount.get("Source", "")).resolve()
        if destination == Path("/workspace"):
            workspace_mount = source
    if workspace_mount != WORKSPACE_HOST_PATH:
        raise PodmanVerificationError("Expected head_pod/workspace to be mounted at /workspace.")

    forbidden_sources = {PRIVATE_CONFIG_PATH, BACKEND_PATH, REPO_ROOT}
    for mount in mounts:
        source = Path(mount.get("Source", "")).resolve()
        if source == WORKSPACE_HOST_PATH:
            continue
        if source in forbidden_sources:
            raise PodmanVerificationError(f"Forbidden mount detected: {source}")
        if BACKEND_PATH in source.parents or PRIVATE_CONFIG_PATH.parent in source.parents:
            raise PodmanVerificationError(f"Forbidden parent mount detected: {source}")

    completed = subprocess.run(
        ["podman", "exec", config.podman_container_name, "test", "-f", "/workspace/AGENT.md"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise PodmanVerificationError("/workspace/AGENT.md is missing inside the container.")



def create_tooling(*, config: AppConfig, container_runner):
    main_client = create_openai_client(config.main_agent.base_url, config.main_agent.api_key)
    sub_client = create_openai_client(config.subagent.base_url, config.subagent.api_key)

    cli_schema, cli_handler = create_cli_tool(container_runner)
    subagent_tools: list[dict] = [cli_schema]
    subagent_handlers: dict[str, Any] = {"run_cli": cli_handler}
    spawn_schema, spawn_handler = create_spawn_subagent_tool(
        openai_client=sub_client,
        model=config.subagent.model,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        tools=subagent_tools,
        tool_handlers=subagent_handlers,
        container_runner=container_runner,
        max_depth=config.maximum_agent_depth,
        max_children=config.maximum_children_per_agent,
    )
    subagent_tools.append(spawn_schema)
    subagent_handlers["spawn_subagent"] = spawn_handler

    main_tools = [cli_schema, spawn_schema]
    main_handlers = {"run_cli": cli_handler, "spawn_subagent": spawn_handler}

    return {
        "main_client": main_client,
        "sub_client": sub_client,
        "main_tools": main_tools,
        "main_handlers": main_handlers,
        "sub_tools": subagent_tools,
        "sub_handlers": subagent_handlers,
    }



def _message_to_incoming_email(row: dict, sender: str) -> mail.IncomingEmail:
    return mail.IncomingEmail(
        from_header=sender,
        subject=row.get("subject") or "",
        content_parts=[],
        message_id=row.get("email_message_id"),
        in_reply_to=row.get("in_reply_to"),
        references=row.get("references_header"),
        timestamp=None,
    )





def validate_config(config: AppConfig) -> None:
    placeholder_values = {
        config.main_agent.api_key,
        config.main_agent.model,
        config.subagent.api_key,
        config.subagent.model,
        config.email_account.password,
        config.imap.password,
        config.smtp.password,
    }
    if any(value.startswith("replace-") for value in placeholder_values):
        raise ValueError("Replace the demonstration values in backhead/private_config.py before running PodHead.")

def bootstrap(config: AppConfig = CONFIG) -> None:
    validate_config(config)
    subprocess.run(["podman", "--version"], check=True, capture_output=True, text=True)
    ensure_state_dir()
    subprocess.run(
        ["podman", "build", "-t", CONTAINER_IMAGE_NAME, "-f", str(REPO_ROOT / "Containerfile"), str(REPO_ROOT)],
        check=True,
    )

    exists = subprocess.run(
        ["podman", "container", "exists", config.podman_container_name],
        check=False,
    )
    if exists.returncode == 0:
        subprocess.run(["podman", "rm", "-f", config.podman_container_name], check=True)

    subprocess.run(
        [
            "podman",
            "create",
            "--name",
            config.podman_container_name,
            "--mount",
            f"type=bind,src={WORKSPACE_HOST_PATH},dst=/workspace",
            CONTAINER_IMAGE_NAME,
        ],
        check=True,
    )
    subprocess.run(["podman", "start", config.podman_container_name], check=True)

    conn = open_db_connection()
    conn.close()

    main_client = create_openai_client(config.main_agent.base_url, config.main_agent.api_key)
    sub_client = create_openai_client(config.subagent.base_url, config.subagent.api_key)
    test_openai_endpoint(main_client, config.main_agent.model)
    test_openai_endpoint(sub_client, config.subagent.model)

    _test_imap(config)
    _test_smtp(config)
    verify_container_environment(config)



def _test_imap(config: AppConfig) -> None:
    client = mail._open_imap_connection(config.imap)
    try:
        client.login(config.imap.username, config.imap.password)
        status, _ = client.select(config.imap.inbox)
        if status != "OK":
            raise RuntimeError(f"Failed to select inbox {config.imap.inbox!r}")
    finally:
        try:
            client.logout()
        except Exception:  # noqa: BLE001
            pass



def _test_smtp(config: AppConfig) -> None:
    with mail.smtplib.SMTP(config.smtp.host, config.smtp.port, timeout=30) as smtp:
        if config.smtp.use_tls:
            smtp.starttls()
        smtp.login(config.smtp.username, config.smtp.password)



async def _process_conversation(
    runtime: RuntimeContext,
    email_thread_id: int,
    run_agent,
    container_runner,
) -> None:
    async with runtime.semaphore:
        while True:
            message_row = await asyncio.to_thread(
                _db_call, runtime, db.claim_next_pending_email_message, email_thread_id
            )
            if message_row is None:
                return
            thread_id = str(email_thread_id)
            sender = await asyncio.to_thread(
                _db_call, runtime, db.get_email_thread_sender, email_thread_id
            )
            history = await asyncio.to_thread(
                _db_call, runtime, db.get_conversation, "email", thread_id
            )
            incoming = _message_to_incoming_email(message_row, sender or "")
            try:
                reply_text = await asyncio.to_thread(run_agent, history, message_row)
                outgoing, outgoing_message_id = mail.build_reply_email(
                    from_address=runtime.config.email_account.address,
                    incoming=incoming,
                    to=sender or "",
                    body=reply_text,
                )
                await asyncio.to_thread(mail.send_reply_smtp, outgoing, runtime.config.smtp)
                ts = db.now_local_iso()
                asst_id = await asyncio.to_thread(
                    _db_call,
                    runtime,
                    db.insert_message_with_content,
                    channel="email",
                    thread_id=thread_id,
                    sender_id="assistant",
                    role="assistant",
                    timestamp=ts,
                    content_parts=[("text", reply_text)],
                )
                await asyncio.to_thread(
                    _db_call,
                    runtime,
                    db.insert_email_message_meta,
                    message_id=asst_id,
                    email_message_id=outgoing_message_id,
                    subject=outgoing.subject,
                    process_state=db.COMPLETED,
                )
                await asyncio.to_thread(
                    _db_call,
                    runtime,
                    db.update_email_message_state,
                    message_row["id"],
                    db.COMPLETED,
                    None,
                )
            except Exception as exc:  # noqa: BLE001
                await asyncio.to_thread(
                    _db_call,
                    runtime,
                    db.update_email_message_state,
                    message_row["id"],
                    db.FAILED,
                    str(exc),
                )
                return


async def schedule_conversation(
    runtime: RuntimeContext, email_thread_id: int, run_agent, container_runner
) -> None:
    async with runtime.processing_tasks_lock:
        existing = runtime.processing_tasks.get(email_thread_id)
        if existing and not existing.done():
            return
        task = asyncio.create_task(
            _process_conversation(runtime, email_thread_id, run_agent, container_runner)
        )
        runtime.processing_tasks[email_thread_id] = task

        def _cleanup(_task, email_thread_id=email_thread_id):
            runtime.processing_tasks.pop(email_thread_id, None)

        task.add_done_callback(_cleanup)


def _poll_inbox_with_db_lock(runtime: RuntimeContext) -> list[dict]:
    with runtime.db_lock:
        return mail.poll_inbox(
            conn=runtime.conn,
            whitelist=runtime.config.sender_whitelist,
            imap_config=runtime.config.imap,
            media_root=MEDIA_ROOT,
        )


async def poll_and_schedule(runtime: RuntimeContext, run_agent, container_runner) -> None:
    results = await asyncio.to_thread(_poll_inbox_with_db_lock, runtime)
    queued_threads = {
        result["email_thread_id"] for result in results if result.get("status") == "queued"
    }
    await asyncio.to_thread(_db_call, runtime, db.requeue_failed_email_messages)
    pending_threads = await asyncio.to_thread(
        _db_call, runtime, db.list_email_threads_with_work
    )
    for email_thread_id in sorted(set(pending_threads) | queued_threads):
        await schedule_conversation(runtime, email_thread_id, run_agent, container_runner)


async def run_backend(config: AppConfig = CONFIG) -> None:
    validate_config(config)
    verify_container_environment(config)
    conn = open_db_connection()
    runtime = RuntimeContext(config, conn)
    await asyncio.to_thread(_db_call, runtime, db.reset_processing_email_messages)
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
    )

    while True:
        await poll_and_schedule(runtime, run_agent, container_runner)
        await asyncio.sleep(config.mail_polling_interval_seconds)



def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the PodHead backend.")
    parser.add_argument("--bootstrap", action="store_true", help="Build and verify the runtime environment.")
    args = parser.parse_args(argv)
    if args.bootstrap:
        bootstrap(CONFIG)
        return
    asyncio.run(run_backend(CONFIG))


if __name__ == "__main__":
    main()
