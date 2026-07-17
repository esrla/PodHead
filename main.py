"""PodHead entry point.

Lifecycle for each incoming email turn:
1. Load private configuration.
2. Create the main and subagent OpenAI clients.
3. Initialize the SQLite database.
4. Initialize the shared container runner.
5. Build backend tool handlers (spawn_subagent, run_cli).
6. For each incoming email, construct a fresh Agent through create_agent(),
   loaded with the prior conversation history from the database.
7. Let the existing mail flow persist and send the returned response.
8. Discard the Agent instance after the turn completes.

No permanent in-memory Agent is kept per email thread.  The database is the
durable source of conversation history.

Limitation: if agent execution or SMTP sending fails after the incoming
message has been inserted, the message will not be retried automatically in
this version.  A restart will detect the already-inserted incoming message as
a duplicate and skip it; manual intervention is needed to clear and retry.
"""

from __future__ import annotations

import sqlite3

from backhead import db
from backhead.agent_loop import DEFAULT_SYSTEM_PROMPT, create_agent, history_to_openai_messages
from backhead.llm import create_openai_client
from backhead.private import load_config
from backhead.tools.cli_tool import create_cli_tool
from backhead.tools.spawn_subagent import create_spawn_subagent_tool


def _fake_container_runner(command: str) -> str:
    """Placeholder container runner used until real Podman execution is wired up."""
    return f"[container runner not yet implemented; command was: {command!r}]"


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
):
    """Return a ``run_agent(history, incoming)`` callback for mail processing.

    The callback converts persisted DB rows into OpenAI messages, strips the
    newest incoming message from prior history (it was inserted before the
    callback fires), constructs an Agent through ``create_agent()``, calls
    ``agent.run(incoming.body)``, and returns the response text.
    """

    def run_agent(history, incoming):
        # ``history`` already contains the newest incoming row at the end.
        # Use all but the last row as prior context so the prompt is not
        # submitted twice.
        prior_rows = history[:-1] if history else []
        prior_messages = history_to_openai_messages(prior_rows)

        agent = create_agent(
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
        return agent.run(incoming.body)

    return run_agent


def main():
    config = load_config()

    agents_cfg = config.get("agents", {})
    limits_cfg = config.get("agent_limits", {})
    max_depth = int(limits_cfg.get("max_depth", 2))
    max_children = int(limits_cfg.get("max_children", 4))

    main_cfg = agents_cfg.get("main", {})
    sub_cfg = agents_cfg.get("subagent", {})

    main_client = create_openai_client(
        base_url=main_cfg.get("base_url", "http://localhost:8080/v1"),
        api_key=main_cfg.get("api_key", "unused"),
    )
    sub_client = create_openai_client(
        base_url=sub_cfg.get("base_url", "http://localhost:8080/v1"),
        api_key=sub_cfg.get("api_key", "unused"),
    )

    conn = sqlite3.connect("state/agent.db")
    db.init_db(conn)

    container_runner = _fake_container_runner

    cli_schema, cli_handler = create_cli_tool(container_runner)

    # Subagent tools (may differ from main-agent tools in a later version).
    subagent_tools = [cli_schema]
    subagent_tool_handlers = {"run_cli": cli_handler}

    spawn_schema, spawn_handler = create_spawn_subagent_tool(
        openai_client=sub_client,
        model=sub_cfg.get("model", "subagent-model"),
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        tools=subagent_tools,
        tool_handlers=subagent_tool_handlers,
        container_runner=container_runner,
        max_depth=max_depth,
        max_children=max_children,
    )

    main_tools = [cli_schema, spawn_schema]
    main_tool_handlers = {"run_cli": cli_handler, "spawn_subagent": spawn_handler}

    whitelist = set(config.get("whitelist", []))

    run_agent = build_email_agent_runner(
        openai_client=main_client,
        model=main_cfg.get("model", "main-model"),
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        tools=main_tools,
        tool_handlers=main_tool_handlers,
        container_runner=container_runner,
        max_depth=max_depth,
        max_children=max_children,
    )

    # Mail processing loop placeholder.
    # Replace with real IMAP polling when email credentials are configured.
    print("[PodHead] Backend initialized. Waiting for incoming email (not yet polled).")
    print(f"[PodHead] Whitelist: {whitelist}")
    _ = conn, run_agent  # referenced to satisfy linter until polling is wired


if __name__ == "__main__":
    main()