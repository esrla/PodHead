# Tests for build_email_agent_runner skill-header sidecar injection.

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import backhead.main as main_module
from backhead.agent_loop import Agent


def _fake_client(reply: str = "done"):
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=reply, tool_calls=[]))]
    )
    return client


def _make_message(text: str, msg_id: int = 1) -> dict:
    return {"id": msg_id, "content": [{"content_type": "text", "content": text}]}


def _capture_system_prompt(monkeypatch) -> dict:
    """Monkeypatch Agent so the system_prompt used in construction is captured."""
    captured = {}
    original_agent = main_module.Agent

    def tracking_agent(**kwargs):
        captured["system_prompt"] = kwargs.get("system_prompt")
        return original_agent(**kwargs)

    monkeypatch.setattr(main_module, "Agent", tracking_agent)
    return captured


# ── skill-header injection ─────────────────────────────────────────────────────


def test_skill_header_is_appended_to_system_prompt(monkeypatch, tmp_path):
    captured = _capture_system_prompt(monkeypatch)
    monkeypatch.setattr(
        main_module,
        "generate_skill_header",
        lambda text, wp, **kw: "- TestSkill: Does testing.",
    )

    run_agent = main_module.build_email_agent_runner(
        openai_client=_fake_client(),
        model="model",
        system_prompt="base-system",
        tools=[],
        tool_handlers={},
        container_runner=None,
        max_depth=1,
        max_children=1,
        workspace_path=tmp_path,
    )
    msg = _make_message("find me a skill")
    run_agent([msg], msg)
    assert "base-system" in captured["system_prompt"]
    assert "TestSkill" in captured["system_prompt"]


def test_system_prompt_unchanged_when_workspace_path_is_none(monkeypatch):
    captured = _capture_system_prompt(monkeypatch)
    should_not_call = [False]

    def guarded_header(*args, **kwargs):
        should_not_call[0] = True
        return None

    monkeypatch.setattr(main_module, "generate_skill_header", guarded_header)

    run_agent = main_module.build_email_agent_runner(
        openai_client=_fake_client(),
        model="model",
        system_prompt="base-system",
        tools=[],
        tool_handlers={},
        container_runner=None,
        max_depth=1,
        max_children=1,
        workspace_path=None,
    )
    msg = _make_message("hello")
    run_agent([msg], msg)
    assert captured["system_prompt"] == "base-system"
    assert not should_not_call[0]


def test_system_prompt_unchanged_when_no_skills_match(monkeypatch, tmp_path):
    captured = _capture_system_prompt(monkeypatch)
    monkeypatch.setattr(main_module, "generate_skill_header", lambda *a, **kw: None)

    run_agent = main_module.build_email_agent_runner(
        openai_client=_fake_client(),
        model="model",
        system_prompt="base-system",
        tools=[],
        tool_handlers={},
        container_runner=None,
        max_depth=1,
        max_children=1,
        workspace_path=tmp_path,
    )
    msg = _make_message("some query")
    run_agent([msg], msg)
    assert captured["system_prompt"] == "base-system"


def test_skill_header_exception_does_not_abort_conversation(monkeypatch, tmp_path):
    captured = _capture_system_prompt(monkeypatch)

    def failing_header(text, wp, **kw):
        raise RuntimeError("skill search failed")

    monkeypatch.setattr(main_module, "generate_skill_header", failing_header)

    run_agent = main_module.build_email_agent_runner(
        openai_client=_fake_client("recovered"),
        model="model",
        system_prompt="base-system",
        tools=[],
        tool_handlers={},
        container_runner=None,
        max_depth=1,
        max_children=1,
        workspace_path=tmp_path,
    )
    msg = _make_message("hello")
    result = run_agent([msg], msg)
    assert result == "recovered"
    assert captured["system_prompt"] == "base-system"


def test_skill_header_not_generated_for_image_only_message(monkeypatch, tmp_path):
    captured = _capture_system_prompt(monkeypatch)
    called_with_text = []

    def tracking_header(text, wp, **kw):
        called_with_text.append(text)
        return None

    monkeypatch.setattr(main_module, "generate_skill_header", tracking_header)

    run_agent = main_module.build_email_agent_runner(
        openai_client=_fake_client(),
        model="model",
        system_prompt="base-system",
        tools=[],
        tool_handlers={},
        container_runner=None,
        max_depth=1,
        max_children=1,
        workspace_path=tmp_path,
    )
    image_only_msg = {"id": 1, "content": [{"content_type": "image", "content": "photo.jpg"}]}
    run_agent([image_only_msg], image_only_msg)
    assert called_with_text == []
    assert captured["system_prompt"] == "base-system"


# ── embed_text tool registration ───────────────────────────────────────────────


def test_create_tooling_registers_embed_text_in_main_and_sub_tools(monkeypatch):
    monkeypatch.setattr(main_module, "create_openai_client", lambda url, key: MagicMock())

    from backhead.private_config import (
        AgentEndpointConfig,
        AppConfig,
        EmailAccountConfig,
        IMAPConfig,
        SMTPConfig,
    )

    config = AppConfig(
        main_agent=AgentEndpointConfig(base_url="http://main", api_key="k", model="m"),
        subagent=AgentEndpointConfig(base_url="http://sub", api_key="k", model="m"),
        email_account=EmailAccountConfig(address="a@b.com", **{"password": "p"}),
        imap=IMAPConfig(host="h", port=993, username="u", **{"password": "p"}, inbox="INBOX", use_ssl=True),
        smtp=SMTPConfig(host="h", port=587, username="u", **{"password": "p"}, use_tls=True),
        sender_whitelist=[],
        mail_polling_interval_seconds=1.0,
        maximum_concurrent_conversations=1,
        maximum_agent_depth=2,
        maximum_children_per_agent=4,
        podman_container_name="test-container",
        workspace_path="head_pod",
        spam_mailbox="Junk",
    )

    tooling = main_module.create_tooling(config=config, container_runner=MagicMock())

    main_names = {t["function"]["name"] for t in tooling["main_tools"]}
    sub_names = {t["function"]["name"] for t in tooling["sub_tools"]}

    assert "embed_text" in main_names
    assert "embed_text" in sub_names
    assert "embed_text" in tooling["main_handlers"]
    assert "embed_text" in tooling["sub_handlers"]
