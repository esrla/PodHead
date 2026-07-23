from __future__ import annotations

import subprocess

from backhead import bootstrap
from backhead import main as main_module
from backhead.private_config import (
    AgentEndpointConfig,
    AppConfig,
    EmailAccountConfig,
    IMAPConfig,
    SMTPConfig,
)


def _config(*, workspace_path: str = "head_pod") -> AppConfig:
    mail_password = "mail-password"
    imap_password = "imap-password"
    smtp_password = "smtp-password"
    return AppConfig(
        main_agent=AgentEndpointConfig(base_url="http://main", api_key="main-key", model="main-model"),
        subagent=AgentEndpointConfig(base_url="http://sub", api_key="sub-key", model="sub-model"),
        email_account=EmailAccountConfig(address="podhead@example.com", **{"password": mail_password}),
        imap=IMAPConfig(
            host="imap.example.com",
            port=993,
            username="imap-user",
            **{"password": imap_password},
            inbox="INBOX",
            use_ssl=True,
        ),
        smtp=SMTPConfig(
            host="smtp.example.com",
            port=587,
            username="smtp-user",
            **{"password": smtp_password},
            use_tls=True,
        ),
        sender_whitelist=["alice@example.com"],
        mail_polling_interval_seconds=15.0,
        maximum_concurrent_conversations=2,
        maximum_agent_depth=2,
        maximum_children_per_agent=4,
        embedding_model="embed-model",
        skill_similarity_threshold=0.35,
        podman_container_name="podhead-agent",
        workspace_path=workspace_path,
        spam_mailbox="Junk",
    )


def _container_inspect(*, running: bool, image_id: str = "image-1", mount_source) -> dict:
    return {
        "Image": image_id,
        "State": {"Running": running},
        "Mounts": [{"Destination": "/workspace", "Source": str(mount_source)}],
    }


def test_main_bootstraps_before_running_backend(monkeypatch):
    calls = []
    sentinel = object()
    monkeypatch.setattr(main_module, "ensure_runtime", lambda config: calls.append(("ensure", config)))
    monkeypatch.setattr(main_module, "run_backend", lambda config: sentinel)
    monkeypatch.setattr(main_module.asyncio, "run", lambda coro: calls.append(("run", coro)))

    main_module.main()

    assert calls == [("ensure", main_module.CONFIG), ("run", sentinel)]


def test_resolve_workspace_host_path_relative_and_absolute(tmp_path):
    relative = _config(workspace_path="head_pod")
    absolute_path = tmp_path / "workspace"
    absolute = _config(workspace_path=str(absolute_path))

    assert bootstrap.resolve_workspace_host_path(relative) == (bootstrap.REPO_ROOT / "head_pod").resolve()
    assert bootstrap.resolve_workspace_host_path(absolute) == absolute_path.resolve()


def test_ensure_container_image_skips_rebuild_when_fingerprint_matches(monkeypatch):
    monkeypatch.setattr(bootstrap, "_runtime_fingerprint", lambda: "expected")
    monkeypatch.setattr(bootstrap, "_podman_exists", lambda kind, name: True)
    monkeypatch.setattr(
        bootstrap,
        "_inspect_image",
        lambda name: {"Id": "image-1", "Labels": {bootstrap.IMAGE_FINGERPRINT_LABEL: "expected"}},
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("image rebuild should be skipped")),
    )

    image_id, rebuilt = bootstrap.ensure_container_image()

    assert (image_id, rebuilt) == ("image-1", False)


def test_runtime_fingerprint_changes_only_for_container_build_inputs(monkeypatch, tmp_path):
    containerfile = tmp_path / "Containerfile"
    container_reqs = tmp_path / "container-requirements.txt"
    backend_reqs = tmp_path / "requirements.txt"
    containerfile.write_text("FROM python:3.12-slim\n")
    container_reqs.write_text("numpy\n")
    backend_reqs.write_text("openai\n")
    monkeypatch.setattr(bootstrap, "IMAGE_BUILD_INPUTS", (containerfile, container_reqs))

    original = bootstrap._runtime_fingerprint()
    backend_reqs.write_text("openai\npytest\n")
    assert bootstrap._runtime_fingerprint() == original

    container_reqs.write_text("numpy\nPyYAML\n")
    assert bootstrap._runtime_fingerprint() != original


def test_ensure_runtime_leaves_valid_running_container_unchanged(monkeypatch, tmp_path):
    calls = []
    config = _config(workspace_path=str(tmp_path / "ws"))
    expected_workspace = bootstrap.resolve_workspace_host_path(config)

    monkeypatch.setattr(bootstrap, "ensure_podman_available", lambda: calls.append("podman"))
    monkeypatch.setattr(bootstrap, "ensure_state_dir", lambda: calls.append("state"))
    monkeypatch.setattr(bootstrap, "ensure_container_image", lambda: ("image-1", False))
    monkeypatch.setattr(bootstrap, "_podman_exists", lambda kind, name: kind == "container")
    monkeypatch.setattr(
        bootstrap,
        "_inspect_container",
        lambda name: _container_inspect(running=True, mount_source=expected_workspace),
    )
    monkeypatch.setattr(bootstrap, "_remove_container", lambda name: calls.append("remove"))
    monkeypatch.setattr(bootstrap, "_create_container", lambda config, workspace: calls.append("create"))
    monkeypatch.setattr(bootstrap, "_start_container", lambda name: calls.append("start"))
    monkeypatch.setattr(bootstrap, "verify_container_environment", lambda config, expected_image_id=None: calls.append("verify"))
    monkeypatch.setattr(bootstrap, "initialize_database", lambda: calls.append("db"))

    bootstrap.ensure_runtime(config)

    assert expected_workspace.exists()
    assert calls == ["podman", "state", "verify", "db"]


def test_ensure_runtime_recreates_container_when_workspace_mount_is_wrong(monkeypatch, tmp_path):
    calls = []
    config = _config(workspace_path=str(tmp_path / "expected"))

    monkeypatch.setattr(bootstrap, "ensure_podman_available", lambda: None)
    monkeypatch.setattr(bootstrap, "ensure_state_dir", lambda: None)
    monkeypatch.setattr(bootstrap, "ensure_container_image", lambda: ("image-1", False))
    monkeypatch.setattr(bootstrap, "_podman_exists", lambda kind, name: kind == "container")
    monkeypatch.setattr(
        bootstrap,
        "_inspect_container",
        lambda name: _container_inspect(running=True, mount_source=tmp_path / "wrong"),
    )
    monkeypatch.setattr(bootstrap, "_remove_container", lambda name: calls.append("remove"))
    monkeypatch.setattr(bootstrap, "_create_container", lambda config, workspace: calls.append(("create", workspace)))
    monkeypatch.setattr(bootstrap, "_start_container", lambda name: calls.append("start"))
    monkeypatch.setattr(bootstrap, "verify_container_environment", lambda config, expected_image_id=None: None)
    monkeypatch.setattr(bootstrap, "initialize_database", lambda: None)

    bootstrap.ensure_runtime(config)

    assert calls == ["remove", ("create", bootstrap.resolve_workspace_host_path(config)), "start"]


def test_verify_container_environment_rejects_wrong_workspace_mount(monkeypatch, tmp_path):
    config = _config(workspace_path=str(tmp_path / "expected"))
    monkeypatch.setattr(
        bootstrap,
        "_inspect_container",
        lambda name: _container_inspect(running=True, mount_source=tmp_path / "wrong"),
    )

    try:
        bootstrap.verify_container_environment(config, expected_image_id="image-1")
    except bootstrap.PodmanVerificationError as exc:
        assert "workspace" in str(exc).lower()
    else:
        raise AssertionError("expected workspace mount verification failure")


def test_ensure_podman_available_explains_installation(monkeypatch):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("podman")

    monkeypatch.setattr(subprocess, "run", fake_run)

    try:
        bootstrap.ensure_podman_available()
    except RuntimeError as exc:
        assert "Install Podman" in str(exc)
        assert "python -m backhead.main" in str(exc)
    else:
        raise AssertionError("expected a runtime error when podman is unavailable")
