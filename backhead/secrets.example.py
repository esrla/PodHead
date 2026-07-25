# Rename this file to secrets.py before running PodHead.
# secrets.py is excluded by .gitignore and must never be committed.
# Replace every demonstration value below with values for your installation.

MAIN_AGENT = {
    "base_url": "http://127.0.0.1:8080/v1",
    "api_key": "replace-main-agent-api-key",
    "model": "replace-main-agent-model",
    "executable_path": "/path/to/llama-server",
    "model_path": "/path/to/main-model.gguf",
    "host": "127.0.0.1",
    "port": 8080,
    "context_size": 2048,
    "threads": 4,
    "embeddings": False,
}

SUBAGENT = {
    "base_url": "http://127.0.0.1:8080/v1",
    "api_key": "replace-subagent-api-key",
    "model": "replace-subagent-model",
    "executable_path": None,
    "model_path": None,
    "host": None,
    "port": None,
    "context_size": None,
    "threads": None,
    "embeddings": False,
}

EMBEDDING_ENDPOINT = {
    "base_url": "http://127.0.0.1:8081/v1",
    "api_key": "replace-embedding-api-key",
    "model": "replace-embedding-model",
    "executable_path": "/path/to/llama-server",
    "model_path": "/path/to/embedding-model.gguf",
    "host": "127.0.0.1",
    "port": 8081,
    "context_size": 2048,
    "threads": 4,
    "embeddings": True,
}

EMAIL_ACCOUNT = {
    "address": "podhead@example.com",
    "password": "replace-email-password",
}

IMAP = {
    "host": "imap.example.com",
    "port": 993,
    "username": "podhead@example.com",
    "password": "replace-imap-password",
    "inbox": "INBOX",
    "use_ssl": True,
}

SMTP = {
    "host": "smtp.example.com",
    "port": 587,
    "username": "podhead@example.com",
    "password": "replace-smtp-password",
    "use_tls": True,
}

SENDER_WHITELIST = [
    "friend@example.com",
]

MAIL_POLLING_INTERVAL_SECONDS = 15.0
MAXIMUM_CONCURRENT_CONVERSATIONS = 2
MAXIMUM_AGENT_DEPTH = 2
MAXIMUM_CHILDREN_PER_AGENT = 4
SKILL_SIMILARITY_THRESHOLD = 0.35
PODMAN_CONTAINER_NAME = "podhead-agent"
WORKSPACE_PATH = "head_pod"
SPAM_MAILBOX = "Junk"
