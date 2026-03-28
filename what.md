# What Program.cs Must Do

Program.cs is the C# backend entry point for PodHead — an always-on lightweight LLM agent
that listens for email triggers and responds via email.

---

## 1. Startup Sequence

1. Load the private configuration from `backend/system.private.json`.
   - If the file does not exist, write a template JSON with all required placeholder values
     and exit with a clear message telling the operator to fill it in before restarting.
   - Validate that every required field is present and non-empty.

2. Ensure the agent container image exists.
   - Run `podman image exists agent-image` (or equivalent).
   - If the image is missing, run `podman build -t agent-image .` and wait for it to finish.

3. Ensure the SQLite database and schema exist.
   - Path: `backend/state/agent.db`
   - Create the `messages` and `summaries` tables if they do not yet exist (see schema below).

---

## 2. Main Loop

Run an infinite polling loop until the process receives a cancellation signal (Ctrl-C / SIGTERM).

Each tick:
1. Call `PollInbox()` to fetch the next unread, whitelisted, deduplicated email event.
   - Resolves the sender to a person via the persons map in config.
   - Skips senders not in the whitelist.
   - Skips emails already recorded in the database (deduplication by `source` + `event_id`).
   - Returns `null` if no qualifying email is waiting.
2. If an event is returned, dispatch it to `AgentLoop.RunTurnAsync(event)`.
3. Wrap the dispatch in a try/catch:
   - On exception: log the error, send a user-friendly error reply email to the original sender,
     and persist an `is_error=1` assistant message in the database.
4. Sleep for `config.Runtime.PollIntervalSec` seconds before the next tick.

---

## 3. Agent Loop (one turn per trigger)

`AgentLoop.RunTurnAsync(event)`:

1. Persist the inbound user message to the `messages` table with `role="user"`,
   `kind="message"`, and `message_reference=NULL`.
2. Build the conversation context for the resolved person:
   - Fetch the current summary from the `summaries` table.
   - Fetch the last `config.Runtime.MaxPairs` question/answer groups, **excluding the current
     turn** (which was just persisted and will be appended as the final user message).
   - Read `head_pod/workspace/preferences.md` and `head_pod/workspace/skills.md`.
3. Compose the LLM message list in this order:
   1. `system` message containing the identity prompt, preferences, and skills.
   2. Prior Q/A history messages (chronological, oldest first).
   3. Current `user` message: summary block (if any) followed by the new user prompt.
4. Call the chat endpoint and enter the tool loop.
5. Persist the assistant response with `role="assistant"`, `kind="message"`,
   and `message_reference=<trigger event_id>`.
6. Update the summary by calling the summarize endpoint and writing it to the `summaries` table.
7. Send the reply email to the triggering sender containing the assistant text.
8. If `SendImageToUser` was invoked, attach the produced image(s) to the reply.

---

## 4. SQLite Schema

```sql
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS summaries (
  person      TEXT    PRIMARY KEY,
  summary     TEXT    NOT NULL DEFAULT '',
  updated_ts  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  person            TEXT    NOT NULL,
  source            TEXT    NOT NULL,
  sender_handle     TEXT    NOT NULL,
  event_id          TEXT    NOT NULL,
  role              TEXT    NOT NULL,
  kind              TEXT    NOT NULL,
  text              TEXT    NOT NULL,
  timestamp         INTEGER NOT NULL,
  message_reference TEXT,
  tool_name         TEXT,
  tool_call_id      TEXT,
  is_error          INTEGER NOT NULL DEFAULT 0,
  embedding         BLOB,
  UNIQUE(source, event_id)
);

CREATE INDEX IF NOT EXISTS idx_messages_person_time ON messages(person, timestamp, id);
CREATE INDEX IF NOT EXISTS idx_messages_person_ref  ON messages(person, message_reference);
CREATE INDEX IF NOT EXISTS idx_messages_tool_call_id ON messages(tool_call_id);
```

---

## 5. Private Config Schema (`backend/system.private.json`)

```json
{
  "email": {
    "imap_host": "",
    "imap_port": 993,
    "imap_user": "",
    "imap_pass": "",
    "smtp_host": "",
    "smtp_user": "",
    "smtp_pass": ""
  },
  "llm": {
    "chat_endpoint": "",
    "embed_endpoint": "",
    "summarize_endpoint": "",
    "api_key": ""
  },
  "whitelist": [],
  "persons": {},
  "runtime": {
    "max_pairs": 10,
    "poll_interval_sec": 30
  }
}
```

---

## 6. Key Constraints

- Every LLM text response is **always** emailed back to the triggering sender — the LLM never
  decides whether an email is sent.
- Agent filesystem (`head_pod/`) must never expose backend files or `system.private.json`.
- The database is append-only; records are never updated or deleted.
- Two consecutive same-role messages must never appear in the LLM prompt.
