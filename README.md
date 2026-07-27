# PodHead

PodHead is an email-driven agent backend with a strict split between backend code and an isolated Podman workspace.

llama.cpp, through a dedicated OpenAI-compatible embedding endpoint, is the only production embedding provider.

## Backend entrypoint

PodHead has one user-facing startup command:

```bash
python -m backhead.main
```

Backend startup bootstraps the runtime automatically before entering the normal mail loop, including health-checking and starting the configured local chat and embedding llama-server processes when needed.

## Configuration

Copy `backhead/secrets.example.py` to `backhead/secrets.py` before running PodHead.
Replace every demonstration value in `backhead/secrets.py`. The local secrets file is excluded by `.gitignore` and must not be committed.

The configuration defines:

- main-agent endpoint and model
- subagent endpoint and model
- dedicated embedding endpoint and model
- managed local model-server settings for chat and embeddings
- skill similarity threshold
- one PodHead email account
- IMAP settings
- SMTP settings
- sender whitelist
- mail polling interval
- maximum concurrent conversations
- maximum agent depth
- maximum children per agent
- Podman container name

## Agent model

`Agent` in `backhead/agent_loop.py` is the only stateful agent class.
Main email handling and `spawn_subagent` both construct agents directly with `Agent(...)`.
Each agent owns its own `self.conversation_history`.
Persisted email history is loaded from SQLite for the main agent, while subagents always start fresh.

## Email processing

- IMAP polling stores incoming whitelisted messages in SQLite.
- Different conversations can run concurrently.
- One conversation is always processed sequentially in FIFO order.
- Message history remains append-only in SQLite.
- Dedicated per-message embeddings and conversation compaction summaries are stored as backend side data only.
- Conversation history is loaded in deterministic `id` order.
- Consecutive user or assistant messages are collapsed before sending history to the model.

## Tool execution

- `spawn_subagent` accepts only `{"prompt": "..."}` from the model.
- The backend chooses each agent's OpenAI-compatible client, model, system prompt, and tools.
- Tool failures are returned to the agent as structured tool-result messages.
- `embed_text`, skill matching, and chat-history indexing/search use the dedicated backend embedding endpoint.
- `search_chat_history` runs entirely in the trusted backend and searches previous conversations for the same sender.
- `run_cli` executes commands only through the configured Podman container.

## Container boundary

The backend verifies that:

- the configured container exists
- the container is running
- `head_pod` is mounted to `/workspace`
- backend code and `backhead/secrets.py` are not mounted into the container
- `/workspace/AGENT.md` exists inside the container

## Installation

Install backend and test dependencies from the repository root with:

```bash
python -m pip install -r requirements.txt
```

Install container-only dependencies inside the image with:

```bash
python -m pip install -r container-requirements.txt
```

## Request failures

- Any exception that prevents a normal reply for the current email request is caught at the outer request boundary.
- The complete original exception, including its full Python stack trace, is returned through the same email thread unchanged.
- If sending that error reply fails, PodHead logs the send failure and continues without retrying the error reply.
- One failed request does not stop the main polling loop or the PodHead process.
- Tool and skill errors that the agent can still handle continue to flow back to the agent as tool results.

## Tests

Run the test suite from the repository root with:

```bash
python -m pytest -q
```

## Admin dashboard

PodHead includes a read-only Streamlit dashboard for inspecting backend activity.

The dashboard shows:

- backend and database status
- recent activity
- conversations and stored message content
- conversation compaction summaries
- embedding coverage
- the PodHead log

Start PodHead and the dashboard together by running:

    ./startup.sh

Streamlit listens only on `127.0.0.1:8501` and is not exposed directly to the local network. When Tailscale is installed and connected, `startup.sh` exposes the dashboard through Tailscale Serve.

Use the HTTPS address printed by `tailscale serve status` to open the dashboard from another device in the same tailnet.

The dashboard reads the SQLite database in read-only mode and does not expose `backhead/secrets.py`.

Streamlit output is written to `~/podhead-web.log`. The PodHead backend continues to write to `~/podhead.log`.
