"""Runtime bootstrap helpers for PodHead backend startup."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import time
from urllib import error as urllib_error
from urllib import request as urllib_request

from backhead import db
from backhead.private_config import CONFIG, AppConfig, AgentEndpointConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = REPO_ROOT / "state"
DB_PATH = STATE_DIR / "agent.db"
CONTAINER_IMAGE_NAME = "podhead-agent-image"
CONTAINERFILE_PATH = REPO_ROOT / "Containerfile"
PRIVATE_CONFIG_PATH = Path(__file__).resolve().parent / "private_config.py"
BACKEND_PATH = (REPO_ROOT / "backhead").resolve()
IMAGE_FINGERPRINT_LABEL = "io.github.esrla.podhead.runtime-fingerprint"
IMAGE_BUILD_INPUTS = (
    CONTAINERFILE_PATH,
    REPO_ROOT / "container-requirements.txt",
)
STARTED_MODEL_SERVER_PROCESSES: list[subprocess.Popen] = []
MODEL_SERVER_STARTUP_TIMEOUT_SECONDS = 15.0
MODEL_SERVER_STARTUP_POLL_INTERVAL_SECONDS = 0.25


class PodmanVerificationError(RuntimeError):
    pass


def ensure_state_dir() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def open_db_connection(path: Path = DB_PATH) -> sqlite3.Connection:
    ensure_state_dir()
    conn = sqlite3.connect(path, check_same_thread=False)
    db.init_db(conn)
    return conn


def validate_config(config: AppConfig) -> None:
    placeholder_values = {
        config.main_agent.api_key,
        config.main_agent.model,
        config.subagent.api_key,
        config.subagent.model,
        config.embedding_endpoint.api_key,
        config.embedding_endpoint.model,
        config.email_account.password,
        config.imap.password,
        config.smtp.password,
    }
    if any(value.startswith("replace-") for value in placeholder_values):
        raise ValueError("Replace the demonstration values in backhead/private_config.py before running PodHead.")
    if not config.workspace_path.strip():
        raise ValueError("workspace_path must not be empty.")
    if not config.spam_mailbox.strip():
        raise ValueError("spam_mailbox must not be empty.")
    if not config.embedding_endpoint.base_url.strip():
        raise ValueError("embedding_endpoint.base_url must not be empty.")
    if not 0.0 <= config.skill_similarity_threshold <= 1.0:
        raise ValueError(
            f"skill_similarity_threshold must be between 0.0 and 1.0, got {config.skill_similarity_threshold}."
        )


def resolve_workspace_host_path(config: AppConfig) -> Path:
    configured = Path(config.workspace_path)
    if configured.is_absolute():
        return configured.resolve()
    return (REPO_ROOT / configured).resolve()


def _runtime_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in IMAGE_BUILD_INPUTS:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def ensure_podman_available() -> None:
    try:
        completed = subprocess.run(
            ["podman", "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Podman is required to run PodHead. Install Podman and then rerun `python -m backhead.main`."
        ) from exc
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(
            f"Podman is required to run PodHead. Install or fix Podman and rerun `python -m backhead.main`."
            + (f" Details: {details}" if details else "")
        )


def _inspect(kind: str, name: str) -> dict:
    completed = subprocess.run(
        ["podman", kind, "inspect", name],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        label = "Image" if kind == "image" else "Container"
        raise PodmanVerificationError(completed.stderr.strip() or f"{label} {name!r} does not exist.")
    items = json.loads(completed.stdout)
    if not items:
        label = "Image" if kind == "image" else "Container"
        raise PodmanVerificationError(f"{label} {name!r} does not exist.")
    return items[0]


def _podman_exists(kind: str, name: str) -> bool:
    completed = subprocess.run(
        ["podman", kind, "exists", name],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _inspect_image(name: str) -> dict:
    return _inspect("image", name)


def _inspect_container(name: str) -> dict:
    return _inspect("container", name)


def ensure_container_image() -> tuple[str, bool]:
    expected_fingerprint = _runtime_fingerprint()
    if _podman_exists("image", CONTAINER_IMAGE_NAME):
        inspect = _inspect_image(CONTAINER_IMAGE_NAME)
        labels = inspect.get("Labels") or {}
        image_id = inspect.get("Id")
        if labels.get(IMAGE_FINGERPRINT_LABEL) == expected_fingerprint and image_id:
            return image_id, False

    subprocess.run(
        [
            "podman",
            "build",
            "--label",
            f"{IMAGE_FINGERPRINT_LABEL}={expected_fingerprint}",
            "-t",
            CONTAINER_IMAGE_NAME,
            "-f",
            str(CONTAINERFILE_PATH),
            str(REPO_ROOT),
        ],
        check=True,
    )
    inspect = _inspect_image(CONTAINER_IMAGE_NAME)
    image_id = inspect.get("Id")
    if not image_id:
        raise PodmanVerificationError(f"Image {CONTAINER_IMAGE_NAME!r} is missing an inspectable image id.")
    return image_id, True


def _workspace_mount_source(inspect: dict) -> Path | None:
    for mount in inspect.get("Mounts") or []:
        destination = Path(mount.get("Destination", ""))
        if destination == Path("/workspace"):
            source = mount.get("Source")
            return Path(source).resolve() if source else None
    return None


def _has_forbidden_mounts(inspect: dict, workspace_host_path: Path) -> bool:
    forbidden_sources = {PRIVATE_CONFIG_PATH, BACKEND_PATH, REPO_ROOT}
    for mount in inspect.get("Mounts") or []:
        source_value = mount.get("Source")
        if not source_value:
            continue
        source = Path(source_value).resolve()
        if source == workspace_host_path:
            continue
        if source in forbidden_sources:
            return True
        if BACKEND_PATH in source.parents or PRIVATE_CONFIG_PATH.parent in source.parents:
            return True
    return False


def _container_requires_recreation(
    inspect: dict, expected_image_id: str, workspace_host_path: Path
) -> bool:
    if inspect.get("Image") != expected_image_id:
        return True
    if _workspace_mount_source(inspect) != workspace_host_path:
        return True
    if _has_forbidden_mounts(inspect, workspace_host_path):
        return True
    return False


def _remove_container(name: str) -> None:
    subprocess.run(["podman", "rm", "-f", name], check=True)


def _create_container(config: AppConfig, workspace_host_path: Path) -> None:
    subprocess.run(
        [
            "podman",
            "create",
            "--name",
            config.podman_container_name,
            "--mount",
            f"type=bind,src={workspace_host_path},dst=/workspace",
            CONTAINER_IMAGE_NAME,
        ],
        check=True,
    )


def _start_container(name: str) -> None:
    subprocess.run(["podman", "start", name], check=True)


def _endpoint_health_url(endpoint: AgentEndpointConfig) -> str:
    return f"{endpoint.base_url.rstrip('/')}/models"


def _is_endpoint_healthy(endpoint: AgentEndpointConfig) -> bool:
    try:
        with urllib_request.urlopen(_endpoint_health_url(endpoint), timeout=2.0) as response:
            return 200 <= getattr(response, "code", 0) < 300
    except (urllib_error.URLError, TimeoutError, ValueError):
        return False


def _expanded_path(path_value: str, *, label: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.exists():
        raise RuntimeError(f"{label} not found: {path}")
    return path


def _validate_managed_endpoint(endpoint: AgentEndpointConfig, *, label: str) -> tuple[Path, Path]:
    required_fields = {
        "executable_path": endpoint.executable_path,
        "model_path": endpoint.model_path,
        "host": endpoint.host,
        "port": endpoint.port,
        "context_size": endpoint.context_size,
        "threads": endpoint.threads,
    }
    missing = [name for name, value in required_fields.items() if value in (None, "")]
    if missing:
        raise RuntimeError(f"{label} is missing managed server configuration: {', '.join(missing)}")
    executable = _expanded_path(str(endpoint.executable_path), label=f"{label} llama-server executable")
    if not executable.is_file():
        raise RuntimeError(f"{label} llama-server executable not found: {executable}")
    model_path = _expanded_path(str(endpoint.model_path), label=f"{label} model file")
    if not model_path.is_file():
        raise RuntimeError(f"{label} model file not found: {model_path}")
    return executable, model_path


def _build_model_server_command(endpoint: AgentEndpointConfig, *, label: str) -> list[str]:
    executable, model_path = _validate_managed_endpoint(endpoint, label=label)
    command = [
        str(executable),
        "--model",
        str(model_path),
        "--ctx-size",
        str(endpoint.context_size),
        "--threads",
        str(endpoint.threads),
    ]
    if endpoint.embeddings:
        command.append("--embeddings")
    command.extend([
        "--host",
        str(endpoint.host),
        "--port",
        str(endpoint.port),
    ])
    return command


def _start_model_server(endpoint: AgentEndpointConfig, *, label: str) -> None:
    command = _build_model_server_command(endpoint, label=label)
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    STARTED_MODEL_SERVER_PROCESSES.append(process)


def _wait_for_endpoint(endpoint: AgentEndpointConfig, *, label: str) -> None:
    deadline = time.monotonic() + MODEL_SERVER_STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if _is_endpoint_healthy(endpoint):
            return
        time.sleep(MODEL_SERVER_STARTUP_POLL_INTERVAL_SECONDS)
    raise RuntimeError(f"{label} did not become healthy at {_endpoint_health_url(endpoint)} after startup.")


def ensure_model_servers(config: AppConfig) -> None:
    managed_endpoints = (
        ("Main model server", config.main_agent),
        ("Embedding model server", config.embedding_endpoint),
    )
    for label, endpoint in managed_endpoints:
        if not endpoint.manages_local_process:
            continue
        if _is_endpoint_healthy(endpoint):
            continue
        _start_model_server(endpoint, label=label)
        _wait_for_endpoint(endpoint, label=label)


def verify_container_environment(config: AppConfig, *, expected_image_id: str | None = None) -> None:
    workspace_host_path = resolve_workspace_host_path(config)
    inspect = _inspect_container(config.podman_container_name)
    state = inspect.get("State") or {}
    if not state.get("Running"):
        raise PodmanVerificationError(f"Container {config.podman_container_name!r} is not running.")
    if expected_image_id is not None and inspect.get("Image") != expected_image_id:
        raise PodmanVerificationError(
            f"Container {config.podman_container_name!r} is not using the expected image."
        )
    if _workspace_mount_source(inspect) != workspace_host_path:
        raise PodmanVerificationError("Expected configured workspace to be mounted at /workspace.")
    if _has_forbidden_mounts(inspect, workspace_host_path):
        raise PodmanVerificationError("Forbidden backend or configuration mount detected.")

    completed = subprocess.run(
        ["podman", "exec", config.podman_container_name, "test", "-f", "/workspace/AGENT.md"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise PodmanVerificationError("/workspace/AGENT.md is missing inside the container.")


def initialize_database() -> None:
    conn = open_db_connection()
    conn.close()


def ensure_runtime(config: AppConfig = CONFIG) -> None:
    validate_config(config)
    ensure_podman_available()
    ensure_state_dir()
    ensure_model_servers(config)
    workspace_host_path = resolve_workspace_host_path(config)
    workspace_host_path.mkdir(parents=True, exist_ok=True)
    image_id, _ = ensure_container_image()

    if _podman_exists("container", config.podman_container_name):
        inspect = _inspect_container(config.podman_container_name)
        if _container_requires_recreation(inspect, image_id, workspace_host_path):
            _remove_container(config.podman_container_name)
            _create_container(config, workspace_host_path)
            _start_container(config.podman_container_name)
        elif not (inspect.get("State") or {}).get("Running"):
            _start_container(config.podman_container_name)
    else:
        _create_container(config, workspace_host_path)
        _start_container(config.podman_container_name)

    verify_container_environment(config, expected_image_id=image_id)
    initialize_database()
