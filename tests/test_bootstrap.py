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


def _config() -> AppConfig:
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
        podman_container_name="podhead-agent",
    )


def _container_inspect(*, running: bool, image_id: str = "image-1", mount_source=None, extra_mounts=None) -> dict:
    mounts = [{"Destination": "/workspace", "Source": str(mount_source or bootstrap.WORKSPACE_HOST_PATH)}]
    mounts.extend(extra_mounts or [])
    return {
        "Image": image_id,
        "State": {"Running": running},
        "Mounts": mounts,
    }


def test_main_bootstraps_before_running_backend(monkeypatch):
    calls = []
    sentinel = object()
    monkeypatch.setattr(main_module, "ensure_runtime", lambda config: calls.append(("ensure", config)))
    monkeypatch.setattr(main_module, "run_backend", lambda config: sentinel)
    monkeypatch.setattr(main_module.asyncio, "run", lambda coro: calls.append(("run", coro)))

    main_module.main()

    assert calls == [("ensure", main_module.CONFIG), ("run", sentinel)]


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


def test_ensure_container_image_rebuilds_when_fingerprint_changes(monkeypatch):
    calls = []
    inspections = iter(
        [
            {"Id": "old-image", "Labels": {bootstrap.IMAGE_FINGERPRINT_LABEL: "stale"}},
            {"Id": "new-image", "Labels": {bootstrap.IMAGE_FINGERPRINT_LABEL: "expected"}},
        ]
    )
    monkeypatch.setattr(bootstrap, "_runtime_fingerprint", lambda: "expected")
    monkeypatch.setattr(bootstrap, "_podman_exists", lambda kind, name: True)
    monkeypatch.setattr(bootstrap, "_inspect_image", lambda name: next(inspections))
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, check: calls.append(command) or subprocess.CompletedProcess(command, 0),
    )

    image_id, rebuilt = bootstrap.ensure_container_image()

    assert (image_id, rebuilt) == ("new-image", True)
    assert calls == [
        [
            "podman",
            "build",
            "--label",
            f"{bootstrap.IMAGE_FINGERPRINT_LABEL}=expected",
            "-t",
            bootstrap.CONTAINER_IMAGE_NAME,
            "-f",
            str(bootstrap.CONTAINERFILE_PATH),
            str(bootstrap.REPO_ROOT),
        ]
    ]


def test_ensure_runtime_leaves_valid_running_container_unchanged(monkeypatch):
    calls = []
    monkeypatch.setattr(bootstrap, "ensure_podman_available", lambda: calls.append("podman"))
    monkeypatch.setattr(bootstrap, "ensure_state_dir", lambda: calls.append("state"))
    monkeypatch.setattr(bootstrap, "ensure_container_image", lambda: ("image-1", False))
    monkeypatch.setattr(bootstrap, "_podman_exists", lambda kind, name: kind == "container")
    monkeypatch.setattr(bootstrap, "_inspect_container", lambda name: _container_inspect(running=True))
    monkeypatch.setattr(bootstrap, "_remove_container", lambda name: calls.append("remove"))
    monkeypatch.setattr(bootstrap, "_create_container", lambda config: calls.append("create"))
    monkeypatch.setattr(bootstrap, "_start_container", lambda name: calls.append("start"))
    monkeypatch.setattr(bootstrap, "verify_container_environment", lambda config, expected_image_id=None: calls.append("verify"))
    monkeypatch.setattr(bootstrap, "initialize_database", lambda: calls.append("db"))
    monkeypatch.setattr(bootstrap, "_verify_external_services", lambda config: calls.append("services"))

    bootstrap.ensure_runtime(_config())

    assert calls == ["podman", "state", "verify", "db", "services"]


def test_ensure_runtime_starts_valid_stopped_container(monkeypatch):
    calls = []
    monkeypatch.setattr(bootstrap, "ensure_podman_available", lambda: None)
    monkeypatch.setattr(bootstrap, "ensure_state_dir", lambda: None)
    monkeypatch.setattr(bootstrap, "ensure_container_image", lambda: ("image-1", False))
    monkeypatch.setattr(bootstrap, "_podman_exists", lambda kind, name: kind == "container")
    monkeypatch.setattr(bootstrap, "_inspect_container", lambda name: _container_inspect(running=False))
    monkeypatch.setattr(bootstrap, "_remove_container", lambda name: calls.append("remove"))
    monkeypatch.setattr(bootstrap, "_create_container", lambda config: calls.append("create"))
    monkeypatch.setattr(bootstrap, "_start_container", lambda name: calls.append("start"))
    monkeypatch.setattr(bootstrap, "verify_container_environment", lambda config, expected_image_id=None: None)
    monkeypatch.setattr(bootstrap, "initialize_database", lambda: None)
    monkeypatch.setattr(bootstrap, "_verify_external_services", lambda config: None)

    bootstrap.ensure_runtime(_config())

    assert calls == ["start"]


def test_ensure_runtime_recreates_container_when_workspace_mount_is_wrong(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(bootstrap, "ensure_podman_available", lambda: None)
    monkeypatch.setattr(bootstrap, "ensure_state_dir", lambda: None)
    monkeypatch.setattr(bootstrap, "ensure_container_image", lambda: ("image-1", False))
    monkeypatch.setattr(bootstrap, "_podman_exists", lambda kind, name: kind == "container")
    monkeypatch.setattr(
        bootstrap,
        "_inspect_container",
        lambda name: _container_inspect(running=True, mount_source=tmp_path / "wrong-workspace"),
    )
    monkeypatch.setattr(bootstrap, "_remove_container", lambda name: calls.append("remove"))
    monkeypatch.setattr(bootstrap, "_create_container", lambda config: calls.append("create"))
    monkeypatch.setattr(bootstrap, "_start_container", lambda name: calls.append("start"))
    monkeypatch.setattr(bootstrap, "verify_container_environment", lambda config, expected_image_id=None: None)
    monkeypatch.setattr(bootstrap, "initialize_database", lambda: None)
    monkeypatch.setattr(bootstrap, "_verify_external_services", lambda config: None)

    bootstrap.ensure_runtime(_config())

    assert calls == ["remove", "create", "start"]


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
