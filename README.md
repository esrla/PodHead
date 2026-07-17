# PodHead
Llm harness ment for my PI, but could be used on any device probably. Yet another llm thing in a container. I just wanted to build my own so I at least trust the code. And no way its getting access to more than i want it to.


## Key Elements Overview

### Introduction
This repository implements an always-on lightweight agent designed specifically for the Raspberry Pi. Its primary purpose is to manage and facilitate various automated tasks efficiently.

### Isolated Framework
The repository features a clear separation between backend and agent environments. The backend handles persistent data management, while the agent operates in a sandboxed environment utilizing `head_pod`. This design ensures that both processes can function optimally without interference.

### Secrets Management
A private JSON configuration file (`private.json`, not committed) is used to securely manage API keys, credentials, and per-agent model configuration.

### SQLite Database
An SQLite database stores explicit conversations and messages. A sender can have multiple
conversations, and each email thread maps to exactly one PodHead conversation.
Email-agent conversation histories are persisted in SQLite; temporary subagent histories
are not persisted in the first version.

## Agent Design

### Agent class
`Agent` (in `backhead/agent_loop.py`) is a reusable stateful conversation object.
One instance owns exactly one conversation history; separate instances never share history.
The official `openai.OpenAI` client is stored as `self.openai_client`.

### Generic agent construction
All agents — both email-triggered and spawned subagents — are created through one
backend helper:

```python
agent = create_agent(
    openai_client=...,
    model=...,
    system_prompt=...,
    conversation_history=...,   # optional prior history
    tools=...,
    tool_handlers=...,
    container_runner=...,
    depth=0,
    max_depth=2,
    max_children=4,
)
```

`main.py` uses this helper for email-triggered agents.
`spawn_subagent` uses the same helper for fresh child contexts.
Parent and child are instances of the same `Agent` class; the only difference
is the configuration supplied when they are constructed.

### Email-agent lifecycle
For each incoming email turn:
1. Prior conversation history is loaded from SQLite and converted to OpenAI messages.
2. A fresh `Agent` is constructed through `create_agent(...)`.
3. The agent is called with the new email body via `agent.run(incoming.body)`.
4. The existing mail flow persists and sends the returned response.
5. The `Agent` instance is discarded; the database is the durable history store.

No permanent in-memory agent is kept per email thread.

### spawn_subagent
The `spawn_subagent` tool lets a parent agent delegate a subtask to a fresh child context.
Tool handlers decide the client's model and configuration for spawned agents.
The calling model supplies only `{"prompt": "..."}` — it cannot provide a model,
endpoint, API key, or system prompt.
The child starts with fresh conversation history; the parent history is not copied.

### Shared workspace
All parent and child agents share one workspace and one Podman container:
- `/workspace/AGENT.md` — mutable workspace guide (inside the container)
- `skills/`, `scripts/`, `memories/`, temporary and output files

OpenAI clients and models may differ between agent instances.

## Conversation Routing Rules
- One shared workspace/container is used for all conversations.
- Sender whitelist validation happens before routing, conversation creation, LLM processing, and replies.
- A new standalone email from a whitelisted sender starts a new conversation.
- A reply with `In-Reply-To`/`References` continues an existing conversation when the referenced
  message belongs to the same normalized sender.
- Conversation history is loaded by conversation ID, so only messages from the resolved
  conversation are sent to the agent.

### Event Listening
The main script (`main.py`) continuously polls an email inbox (example provided) to monitor incoming events. The workflow comprises scanning, validating, and triggering deployments based on events that have been authenticated via the backend.

### Tools and CLI
The repository distinguishes between explicit LLM tools (e.g. `spawn_subagent`, `run_cli`)
and general-purpose scripts. Tool schemas are backend-owned; the model cannot change tool
configuration. CLI commands execute inside the containerized environment.

### Container Editing
The filesystem of the container (`head_pod`) is accessible for manual adjustments, offering flexibility for users while ensuring that it remains isolated from the backend code. This approach upholds security standards while allowing customization.