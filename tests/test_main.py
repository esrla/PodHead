from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import backhead.main as main_module


def _fake_chat_client(reply: str = "done"):
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=reply, tool_calls=[]))]
    )
    return client


def _embedding_response(vector):
    return MagicMock(data=[MagicMock(index=0, embedding=vector)])


def _make_message(text: str, msg_id: int = 1) -> dict:
    return {
        "id": msg_id,
        "thread_id": "thread-1",
        "sender_id": "alice@example.com",
        "content": [{"content_type": "text", "content": text}],
    }


def _config():
    return SimpleNamespace(
        MAIN_AGENT={
            "base_url": "http://main",
            "api_key": "main-key",
            "model": "main-model",
        },
        SUBAGENT={
            "base_url": "http://sub",
            "api_key": "sub-key",
            "model": "sub-model",
        },
        EMBEDDING_ENDPOINT={
            "base_url": "http://embed",
            "api_key": "embed-key",
            "model": "embed-model",
        },
        EMAIL_ACCOUNT={
            "address": "podhead@example.com",
            "password": "mail-password",
        },
        IMAP={
            "host": "imap.example.com",
            "port": 993,
            "username": "imap-user",
            "password": "imap-password",
            "inbox": "INBOX",
            "use_ssl": True,
        },
        SMTP={
            "host": "smtp.example.com",
            "port": 587,
            "username": "smtp-user",
            "password": "smtp-password",
            "use_tls": True,
        },
        SENDER_WHITELIST=[],
        MAIL_POLLING_INTERVAL_SECONDS=1.0,
        MAXIMUM_CONCURRENT_CONVERSATIONS=1,
        MAXIMUM_AGENT_DEPTH=2,
        MAXIMUM_CHILDREN_PER_AGENT=4,
        SKILL_SIMILARITY_THRESHOLD=0.35,
        PODMAN_CONTAINER_NAME="test-container",
        WORKSPACE_PATH="head_pod",
        SPAM_MAILBOX="Junk",
    )


def test_build_email_agent_runner_passes_workspace_skill_provider_and_backend_context_to_agent(
    monkeypatch,
    tmp_path,
):
    captured = {}
    original_agent = main_module.Agent

    def tracking_agent(**kwargs):
        captured["system_prompt"] = kwargs["system_prompt"]
        captured["workspace_path"] = kwargs.get("workspace_path")
        captured["skill_header_provider"] = kwargs.get("skill_header_provider")
        captured["backend_context"] = kwargs.get("backend_context")
        return original_agent(**kwargs)

    provider = MagicMock(return_value=None)
    monkeypatch.setattr(main_module, "Agent", tracking_agent)

    run_agent = main_module.build_email_agent_runner(
        openai_client=_fake_chat_client(),
        model="model",
        system_prompt="base-system",
        tools=[],
        tool_handlers={},
        container_runner=None,
        max_depth=1,
        max_children=1,
        workspace_path=tmp_path,
        skill_header_provider=provider,
    )
    message = _make_message("find me a skill")
    run_agent([message], message)

    assert captured["system_prompt"] == "base-system"
    assert captured["workspace_path"] == tmp_path
    assert captured["skill_header_provider"] is provider
    assert captured["backend_context"] == {
        "sender_id": "alice@example.com",
        "thread_id": "thread-1",
    }
    provider.assert_called_once_with("find me a skill", tmp_path)


def test_build_email_agent_runner_skips_skill_provider_for_image_only_message(
    tmp_path,
):
    provider = MagicMock(return_value=None)
    run_agent = main_module.build_email_agent_runner(
        openai_client=_fake_chat_client(),
        model="model",
        system_prompt="base-system",
        tools=[],
        tool_handlers={},
        container_runner=None,
        max_depth=1,
        max_children=1,
        workspace_path=tmp_path,
        skill_header_provider=provider,
    )
    image_only_msg = {
        "id": 1,
        "thread_id": "thread-1",
        "sender_id": "alice@example.com",
        "content": [{"content_type": "image", "content": "photo.jpg"}],
    }

    run_agent([image_only_msg], image_only_msg)

    provider.assert_called_once_with("[image: photo.jpg]", tmp_path)


def test_create_tooling_registers_search_and_embed_tools_in_main_and_sub_tools(
    monkeypatch,
):
    created_clients = [
        _fake_chat_client(),
        _fake_chat_client(),
        MagicMock(),
    ]
    monkeypatch.setattr(
        main_module,
        "create_openai_client",
        lambda url, key: created_clients.pop(0),
    )

    tooling = main_module.create_tooling(
        config=_config(),
        container_runner=MagicMock(),
    )

    main_names = {
        tool["function"]["name"]
        for tool in tooling["main_tools"]
    }
    sub_names = {
        tool["function"]["name"]
        for tool in tooling["sub_tools"]
    }

    assert "embed_text" in main_names
    assert "embed_text" in sub_names
    assert "search_chat_history" in main_names
    assert "search_chat_history" in sub_names
    assert "embed_text" in tooling["main_handlers"]
    assert "embed_text" in tooling["sub_handlers"]
    assert "search_chat_history" in tooling["main_handlers"]
    assert "search_chat_history" in tooling["sub_handlers"]


def test_create_tooling_uses_dedicated_embedding_api_for_embed_text_and_skill_headers(
    monkeypatch,
    tmp_path,
):
    main_client = _fake_chat_client()
    sub_client = _fake_chat_client()
    embedding_client = MagicMock()
    embedding_client.embeddings.create.side_effect = [
        _embedding_response([1.0, 0.0]),
        _embedding_response([1.0, 0.0]),
        _embedding_response([1.0, 0.0]),
    ]

    created_clients = [
        main_client,
        sub_client,
        embedding_client,
    ]
    monkeypatch.setattr(
        main_module,
        "create_openai_client",
        lambda url, key: created_clients.pop(0),
    )

    skill_dir = tmp_path / "skills" / "calc"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: Calculator\n"
        "description: Does arithmetic.\n"
        "---\n"
        "# Body"
    )

    tooling = main_module.create_tooling(
        config=_config(),
        container_runner=MagicMock(),
    )
    embed_result = tooling["main_handlers"]["embed_text"](
        {"text": "hello world"},
        None,
    )
    header = tooling["skill_header_provider"](
        "arithmetic please",
        tmp_path,
    )

    assert embed_result == {
        "ok": True,
        "embedding": [1.0, 0.0],
    }
    assert header is not None
    assert "[Relevant skills]" in header
    assert "Calculator" in header
    assert embedding_client.embeddings.create.call_args_list == [
        call(
            model="embed-model",
            input=["title: none | text: hello world"],
        ),
        call(
            model="embed-model",
            input=["task: search result | query: arithmetic please"],
        ),
        call(
            model="embed-model",
            input=["title: none | text: Calculator. Does arithmetic."],
        ),
    ]


def test_build_email_agent_runner_uses_history_builder(tmp_path):
    provider = MagicMock(return_value=None)
    history_builder = MagicMock(
        return_value=[{"role": "system", "content": "summary"}]
    )
    run_agent = main_module.build_email_agent_runner(
        openai_client=_fake_chat_client(),
        model="model",
        system_prompt="base-system",
        tools=[],
        tool_handlers={},
        container_runner=None,
        max_depth=1,
        max_children=1,
        workspace_path=tmp_path,
        skill_header_provider=provider,
        history_builder=history_builder,
    )
    message = _make_message("hello")

    run_agent([message], message)

    history_builder.assert_called_once_with([], message)
