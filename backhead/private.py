"""Private backend configuration loader.

Reads ``private.json`` from the repository root.  If the file does not exist
a template is written and the process exits with instructions.

The file is excluded from Git; never commit real credentials.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "private.json"
)

_TEMPLATE: dict[str, Any] = {
    "agents": {
        "main": {
            "base_url": "http://localhost:8080/v1",
            "api_key": "unused",
            "model": "main-model",
        },
        "subagent": {
            "base_url": "http://localhost:8080/v1",
            "api_key": "unused",
            "model": "subagent-model",
        },
    },
    "agent_limits": {
        "max_depth": 2,
        "max_children": 4,
    },
    "email": {
        "imap_host": "imap.example.com",
        "imap_user": "user@example.com",
        "imap_pass": "changeme",
        "smtp_host": "smtp.example.com",
        "smtp_user": "user@example.com",
        "smtp_pass": "changeme",
    },
    "whitelist": ["user@example.com"],
}


def load_config(path: str | None = None) -> dict[str, Any]:
    """Load and return the private configuration dictionary.

    Creates a template ``private.json`` and exits if the file is missing.
    """
    config_path = path or _DEFAULT_CONFIG_PATH
    if not os.path.exists(config_path):
        with open(config_path, "w") as fh:
            json.dump(_TEMPLATE, fh, indent=2)
        print(
            f"[PodHead] Created template config at {config_path}. "
            "Fill in your credentials and restart.",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(config_path) as fh:
        return json.load(fh)