"""Runtime bootstrap helpers for PodHead backend startup."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess

from backhead import db, mail
from backhead.llm import create_openai_client, test_openai_endpoint
from backhead.private_config import CONFIG, AppConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = REPO_ROOT / "state"
DB_PATH = STATE_DIR / "agent.db"
CONTAINER_IMAGE_NAME = "podhead-agent-image"
WORKSPACE_HOST_PATH = (REPO_ROOT / "head_pod").resolve()
CONTAINERFILE_PATH = REPO_ROOT / "Containerfile"
PRIVATE_CONFIG_PATH = Path(__file__).resolve().parent / "private_config.py"
BACKEND_PATH = (REPO_ROOT / "backhead").resolve()
IMAGE_FINGERPRINT_LABEL = "io.github.esrla.podhead.runtime-fingerprint"
IMAGE_BUILD_INPUTS = (
    CONTAINERFILE_PATH,
    REPO_ROOT / "requirements.txt",
)


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
        config.email_account.password,
        config.imap.password,
        config.smtp.password,
    }
    if any(value.startswith("replace-") for value in placeholder_values):
        raise ValueError("Replace the demonstration values in backhead/private_config.py before running PodHead.")


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


def _has_forbidden_mounts(inspect: dict) -> bool:
    forbidden_sources = {PRIVATE_CONFIG_PATH, BACKEND_PATH, REPO_ROOT}
    for mount in inspect.get("Mounts") or []:
        source_value = mount.get("Source")
        if not source_value:
            continue
        source = Path(source_value).resolve()
        if source == WORKSPACE_HOST_PATH:
            continue
        if source in forbidden_sources:
            return True
        if BACKEND_PATH in source.parents or PRIVATE_CONFIG_PATH.parent in source.parents:
            return True
    return False


def _container_requires_recreation(inspect: dict, expected_image_id: str) -> bool:
    if inspect.get("Image") != expected_image_id:
        return True
    if _workspace_mount_source(inspect) != WORKSPACE_HOST_PATH:
        return True
    if _has_forbidden_mounts(inspect):
        return True
    return False


def _remove_container(name: str) -> None:
    subprocess.run(["podman", "rm", "-f", name], check=True)


def _create_container(config: AppConfig) -> None:
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


def _start_container(name: str) -> None:
    subprocess.run(["podman", "start", name], check=True)


def verify_container_environment(config: AppConfig, *, expected_image_id: str | None = None) -> None:
    inspect = _inspect_container(config.podman_container_name)
    state = inspect.get("State") or {}
    if not state.get("Running"):
        raise PodmanVerificationError(f"Container {config.podman_container_name!r} is not running.")
    if expected_image_id is not None and inspect.get("Image") != expected_image_id:
        raise PodmanVerificationError(
            f"Container {config.podman_container_name!r} is not using the expected image."
        )
    if _workspace_mount_source(inspect) != WORKSPACE_HOST_PATH:
        raise PodmanVerificationError("Expected head_pod to be mounted at /workspace.")
    if _has_forbidden_mounts(inspect):
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


def _verify_external_services(config: AppConfig) -> None:
    main_client = create_openai_client(config.main_agent.base_url, config.main_agent.api_key)
    sub_client = create_openai_client(config.subagent.base_url, config.subagent.api_key)
    test_openai_endpoint(main_client, config.main_agent.model)
    test_openai_endpoint(sub_client, config.subagent.model)
    _test_imap(config)
    _test_smtp(config)


def ensure_runtime(config: AppConfig = CONFIG) -> None:
    validate_config(config)
    ensure_podman_available()
    ensure_state_dir()
    image_id, _ = ensure_container_image()

    if _podman_exists("container", config.podman_container_name):
        inspect = _inspect_container(config.podman_container_name)
        if _container_requires_recreation(inspect, image_id):
            _remove_container(config.podman_container_name)
            _create_container(config)
            _start_container(config.podman_container_name)
        elif not (inspect.get("State") or {}).get("Running"):
            _start_container(config.podman_container_name)
    else:
        _create_container(config)
        _start_container(config.podman_container_name)

    verify_container_environment(config, expected_image_id=image_id)
    initialize_database()
    _verify_external_services(config)
