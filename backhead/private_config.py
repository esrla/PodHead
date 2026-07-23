"""Private runtime configuration.

Replace the demonstration values before running PodHead.
Do not commit real secrets to the repository.
"""

from __future__ import annotations


class AgentEndpointConfig:
    def __init__(self, *, base_url: str, api_key: str, model: str) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self._model = model

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def model(self) -> str:
        return self._model


class EmailAccountConfig:
    def __init__(self, *, address: str, password: str) -> None:
        self._address = address
        self._password = password

    @property
    def address(self) -> str:
        return self._address

    @property
    def password(self) -> str:
        return self._password


class IMAPConfig:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        username: str,
        password: str,
        inbox: str,
        use_ssl: bool,
    ) -> None:
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._inbox = inbox
        self._use_ssl = use_ssl

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    @property
    def username(self) -> str:
        return self._username

    @property
    def password(self) -> str:
        return self._password

    @property
    def inbox(self) -> str:
        return self._inbox

    @property
    def use_ssl(self) -> bool:
        return self._use_ssl


class SMTPConfig:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        username: str,
        password: str,
        use_tls: bool,
    ) -> None:
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._use_tls = use_tls

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    @property
    def username(self) -> str:
        return self._username

    @property
    def password(self) -> str:
        return self._password

    @property
    def use_tls(self) -> bool:
        return self._use_tls


class AppConfig:
    def __init__(
        self,
        *,
        main_agent: AgentEndpointConfig,
        subagent: AgentEndpointConfig,
        email_account: EmailAccountConfig,
        imap: IMAPConfig,
        smtp: SMTPConfig,
        sender_whitelist: list[str],
        mail_polling_interval_seconds: float,
        maximum_concurrent_conversations: int,
        maximum_agent_depth: int,
        maximum_children_per_agent: int,
        embedding_model: str,
        skill_similarity_threshold: float,
        podman_container_name: str,
        workspace_path: str,
        spam_mailbox: str,
    ) -> None:
        self._main_agent = main_agent
        self._subagent = subagent
        self._email_account = email_account
        self._imap = imap
        self._smtp = smtp
        self._sender_whitelist = list(sender_whitelist)
        self._mail_polling_interval_seconds = mail_polling_interval_seconds
        self._maximum_concurrent_conversations = maximum_concurrent_conversations
        self._maximum_agent_depth = maximum_agent_depth
        self._maximum_children_per_agent = maximum_children_per_agent
        self._embedding_model = embedding_model
        self._skill_similarity_threshold = skill_similarity_threshold
        self._podman_container_name = podman_container_name
        self._workspace_path = workspace_path
        self._spam_mailbox = spam_mailbox

    @property
    def main_agent(self) -> AgentEndpointConfig:
        return self._main_agent

    @property
    def subagent(self) -> AgentEndpointConfig:
        return self._subagent

    @property
    def email_account(self) -> EmailAccountConfig:
        return self._email_account

    @property
    def imap(self) -> IMAPConfig:
        return self._imap

    @property
    def smtp(self) -> SMTPConfig:
        return self._smtp

    @property
    def sender_whitelist(self) -> list[str]:
        return list(self._sender_whitelist)

    @property
    def mail_polling_interval_seconds(self) -> float:
        return self._mail_polling_interval_seconds

    @property
    def maximum_concurrent_conversations(self) -> int:
        return self._maximum_concurrent_conversations

    @property
    def maximum_agent_depth(self) -> int:
        return self._maximum_agent_depth

    @property
    def maximum_children_per_agent(self) -> int:
        return self._maximum_children_per_agent

    @property
    def embedding_model(self) -> str:
        return self._embedding_model

    @property
    def skill_similarity_threshold(self) -> float:
        return self._skill_similarity_threshold

    @property
    def podman_container_name(self) -> str:
        return self._podman_container_name

    @property
    def workspace_path(self) -> str:
        return self._workspace_path

    @property
    def spam_mailbox(self) -> str:
        return self._spam_mailbox


CONFIG = AppConfig(
    main_agent=AgentEndpointConfig(
        base_url="http://127.0.0.1:11434/v1",
        api_key="replace-main-agent-api-key",
        model="replace-main-agent-model",
    ),
    subagent=AgentEndpointConfig(
        base_url="http://127.0.0.1:11435/v1",
        api_key="replace-subagent-api-key",
        model="replace-subagent-model",
    ),
    email_account=EmailAccountConfig(
        address="podhead@example.com",
        password="replace-email-account-password",
    ),
    imap=IMAPConfig(
        host="imap.example.com",
        port=993,
        username="podhead@example.com",
        password="replace-imap-password",
        inbox="INBOX",
        use_ssl=True,
    ),
    smtp=SMTPConfig(
        host="smtp.example.com",
        port=587,
        username="podhead@example.com",
        password="replace-smtp-password",
        use_tls=True,
    ),
    sender_whitelist=["friend@example.com"],
    mail_polling_interval_seconds=15.0,
    maximum_concurrent_conversations=2,
    maximum_agent_depth=2,
    maximum_children_per_agent=4,
    embedding_model="replace-embedding-model",
    skill_similarity_threshold=0.35,
    podman_container_name="podhead-agent",
    workspace_path="head_pod",
    spam_mailbox="Junk",
)
