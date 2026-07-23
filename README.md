# PodHead

PodHead is an email-driven agent backend with a strict split between backend code and an isolated Podman workspace.

llama.cpp, through its OpenAI-compatible embeddings endpoint, is the only production embedding provider.

## Backend entrypoint

PodHead has one user-facing startup command:

```bash
python -m backhead.main
```

Backend startup bootstraps the runtime automatically before entering the normal mail loop.

## Configuration

Edit `/home/runner/work/PodHead/PodHead/backhead/private_config.py` before running PodHead.
The committed values are demonstrations only. Replace them with your own values and do not commit real secrets.

The configuration defines:

- main-agent endpoint and model
- subagent endpoint and model
- llama.cpp-compatible embedding model
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

- IMAP polling stores incoming whitelisted messages in SQLite with persistent processing states.
- Different conversations can run concurrently.
- One conversation is always processed sequentially in FIFO order.
- Failed incoming messages stay in SQLite and are retried by the backend loop.
- Conversation history is loaded in deterministic `timestamp, id` order.
- Consecutive user or assistant messages are collapsed before sending history to the model.

## Tool execution

- `spawn_subagent` accepts only `{"prompt": "..."}` from the model.
- The backend chooses each agent's OpenAI-compatible client, model, system prompt, and tools.
- Tool failures are returned to the agent as structured tool-result messages.
- `embed_text` and skill matching use the configured llama.cpp/OpenAI-compatible embeddings endpoint and embedding model.
- `run_cli` executes commands only through the configured Podman container.

## Container boundary

The backend verifies that:

- the configured container exists
- the container is running
- `head_pod` is mounted to `/workspace`
- backend code and `backhead/private_config.py` are not mounted into the container
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
