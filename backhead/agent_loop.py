"""Agent loop, execution-tree tracking, and message-history conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from backhead.media import get_image_mime_type, load_image_as_base64

AGENT_WORKSPACE_GUIDE = "/workspace/AGENT.md"
HISTORY_SEPARATOR = "\n\n---\n\n"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant running inside PodHead. "
    f"At the start of each session read {AGENT_WORKSPACE_GUIDE} for workspace instructions. "
    "Follow the instructions found there for all tool use and file management. "
    "Relevant skills may appear here as headers only; read the referenced /workspace/skills/.../SKILL.md before using one, follow it, do not infer the skill body from the name or description alone, and treat /workspace as your persistent workspace root."
)


def tool_error_result(
    error_type: str,
    message: str,
    details: str | None = None,
) -> dict[str, Any]:
    """Return a structured tool-error payload for the agent."""
    return {
        "ok": False,
        "error": {
            "type": error_type,
            "message": message,
            "details": details or "",
        },
    }


@dataclass
class ToolRecord:
    name: str
    args_display: str
    result_preview: str
    child_agent: "Agent | None" = None


def _preview(text: str, max_chars: int = 100) -> str:
    """Return preview of text, truncated and marked if needed."""
    text = re.sub(
        r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+",
        "[image base64 omitted]",
        text,
    )
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... [truncated]"


def _sanitize_for_preview(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return "[binary data omitted]"
    if isinstance(value, dict):
        return {k: _sanitize_for_preview(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_preview(v) for v in value]
    return value


def _format_args_display(args_json: str) -> str:
    return args_json


def _format_result_preview(result: Any) -> str:
    if isinstance(result, (bytes, bytearray)):
        text = "[binary data omitted]"
    elif isinstance(result, (dict, list)):
        text = json.dumps(_sanitize_for_preview(result))
    else:
        text = str(result)
    return _preview(text)


def _prompt_text_for_skill_matching(prompt: str | list) -> str:
    if isinstance(prompt, str):
        return prompt.strip()
    if isinstance(prompt, list):
        text_parts = []
        for item in prompt:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "".join(text_parts).strip()
    return ""


def _build_tree_lines(
    agent_label: str,
    tool_records: list[ToolRecord],
    indent: str = "",
    prefix: str = "",
) -> list[str]:
    """Build execution tree lines for one agent level."""
    lines = [f"{prefix}{agent_label}"]
    total = len(tool_records)
    for i, rec in enumerate(tool_records):
        is_last = i == total - 1
        connector = "└──" if is_last else "├──"
        child_indent = indent + ("    " if is_last else "│   ")
        lines.append(f"{indent}{connector} Tool call: {rec.name}({rec.args_display})")

        if rec.child_agent is not None:
            child_label_indent = child_indent
            child_records = rec.child_agent._tool_records
            child_reply_preview = _preview(rec.child_agent._final_reply or "")
            child_lines = _build_tree_lines(
                "Subagent",
                child_records,
                indent=child_label_indent + "    ",
                prefix=child_label_indent,
            )
            for cl in child_lines:
                lines.append(cl)
            if child_reply_preview:
                lines.append(f"{child_label_indent}    └── Reply: {child_reply_preview}")
        else:
            lines.append(f"{child_indent}└── Tool result: {rec.result_preview}")

    return lines


def build_execution_tree(agent: "Agent", final_reply: str) -> str | None:
    """Build the execution tree text if tools were used, else return None."""
    if not agent._tool_records:
        return None
    lines = _build_tree_lines("Main agent", agent._tool_records)
    return "[System-generated execution tree]\n" + "\n".join(lines)


# --------------------------------------------------------------------------- #
# Message-history conversion
# --------------------------------------------------------------------------- #


def messages_to_openai_messages(
    conversation: list[dict],
    media_root: Path | None = None,
) -> list[dict]:
    """Convert generic stored messages to OpenAI Chat Completions format.

    Consecutive same-role messages are merged with ``HISTORY_SEPARATOR`` between
    them. Content is a plain string for text-only messages, or a list of content
    dicts when any image part is present.
    """
    processed: list[dict] = []
    for stored_msg in conversation:
        role = stored_msg["role"]
        parts = stored_msg.get("content", [])
        oai_parts: list[dict] = []
        for part in parts:
            ct = part["content_type"]
            val = part["content"]
            if ct == "text":
                oai_parts.append({"type": "text", "text": val})
            elif ct == "image":
                if media_root:
                    b64 = load_image_as_base64(val, media_root)
                    if b64:
                        mime = get_image_mime_type(val)
                        oai_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{b64}"},
                            }
                        )
                    else:
                        oai_parts.append({"type": "text", "text": "[image not found]"})
                else:
                    oai_parts.append({"type": "text", "text": f"[image: {val}]"})
        if oai_parts:
            processed.append({"role": role, "parts": oai_parts})

    merged: list[dict] = []
    for item in processed:
        if merged and merged[-1]["role"] == item["role"]:
            merged[-1]["parts"].append({"type": "text", "text": HISTORY_SEPARATOR})
            merged[-1]["parts"].extend(item["parts"])
        else:
            merged.append({"role": item["role"], "parts": list(item["parts"])})

    result: list[dict] = []
    for item in merged:
        parts = item["parts"]
        if all(p["type"] == "text" for p in parts):
            content: str | list = "".join(p["text"] for p in parts)
        else:
            content = parts
        result.append({"role": item["role"], "content": content})

    return result


class Agent:
    """Stateful conversation context bound to one set of backend dependencies."""

    def __init__(
        self,
        *,
        openai_client: Any,
        model: str,
        system_prompt: str,
        workspace_path: Path | None = None,
        skill_header_provider: Any = None,
        conversation_history: list[dict] | None = None,
        tools: list[dict] | None = None,
        tool_handlers: dict[str, Any] | None = None,
        container_runner: Any = None,
        depth: int = 0,
        max_depth: int = 2,
        max_children: int = 4,
        backend_context: dict[str, Any] | None = None,
    ) -> None:
        self.openai_client = openai_client
        self.model = model
        self.system_prompt = system_prompt
        self.workspace_path = workspace_path
        self.skill_header_provider = skill_header_provider
        self.conversation_history: list[dict] = list(conversation_history or [])
        self.tools: list[dict] = list(tools or [])
        self.tool_handlers: dict[str, Any] = dict(tool_handlers or {})
        self.container_runner = container_runner
        self.depth = depth
        self.max_depth = max_depth
        self.max_children = max_children
        self.backend_context: dict[str, Any] = dict(backend_context or {})
        self._tool_records: list[ToolRecord] = []
        self._final_reply: str = ""
        self._children_spawned = 0
        self._pending_child_agent: "Agent | None" = None
        self._skills_injected = False

    def _inject_relevant_skills_once(self, prompt: str | list) -> None:
        if self._skills_injected:
            return
        self._skills_injected = True
        if self.workspace_path is None or self.skill_header_provider is None:
            return
        prompt_text = _prompt_text_for_skill_matching(prompt)
        if not prompt_text:
            return
        header = self.skill_header_provider(prompt_text, self.workspace_path)
        if header:
            self.system_prompt = f"{self.system_prompt}\n\n{header}"

    def _append_tool_result(self, tool_call_id: str, result: Any) -> None:
        if isinstance(result, (dict, list)):
            content = json.dumps(_sanitize_for_preview(result))
        elif isinstance(result, (bytes, bytearray)):
            content = "[binary data omitted]"
        else:
            content = str(result)
        self.conversation_history.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
            }
        )

    def run(self, prompt: str | list) -> str:
        """Append *prompt* as a user message and run the agent loop to completion."""
        self._inject_relevant_skills_once(prompt)
        self.conversation_history.append({"role": "user", "content": prompt})
        return self._run_loop()

    def _run_loop(self) -> str:
        while True:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    *self.conversation_history,
                ],
            }
            if self.tools:
                kwargs["tools"] = self.tools
                kwargs["tool_choice"] = "auto"

            response = self.openai_client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            msg = choice.message

            assistant_turn: dict[str, Any] = {
                "role": "assistant",
                "content": msg.content,
            }
            if msg.tool_calls:
                assistant_turn["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            self.conversation_history.append(assistant_turn)

            if not msg.tool_calls:
                final_reply = msg.content or ""
                self._final_reply = final_reply
                if self.depth == 0:
                    print("#agent reply", flush=True)
                    tree = build_execution_tree(self, final_reply)
                    if tree:
                        return f"{tree}\n---\n{final_reply}"
                return final_reply

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                handler = self.tool_handlers.get(tool_name)
                if handler is None:
                    result = tool_error_result(
                        "tool_execution_error",
                        f"Unknown tool '{tool_name}'.",
                        "No backend handler is registered for this tool.",
                    )
                    self._record_tool(tool_name, tc.function.arguments, result, None)
                    self._append_tool_result(tc.id, result)
                    continue

                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError as exc:
                    result = tool_error_result(
                        "tool_execution_error",
                        "Invalid JSON arguments.",
                        str(exc),
                    )
                    self._record_tool(tool_name, tc.function.arguments, result, None)
                    self._append_tool_result(tc.id, result)
                    continue

                try:
                    print(f"#agent tool tool={tool_name}", flush=True)
                    result = handler(args, self)
                except TimeoutError as exc:
                    result = tool_error_result(
                        "tool_execution_error",
                        "Tool timed out.",
                        str(exc),
                    )
                except Exception as exc:  # noqa: BLE001
                    error_type = getattr(exc, "error_type", "tool_execution_error")
                    result = tool_error_result(
                        error_type,
                        "Tool execution failed.",
                        str(exc),
                    )

                child_agent = self._pending_child_agent
                self._pending_child_agent = None
                self._record_tool(tool_name, tc.function.arguments, result, child_agent)
                self._append_tool_result(tc.id, result)

    def _record_tool(
        self,
        tool_name: str,
        args_json: str,
        result: Any,
        child_agent: "Agent | None",
    ) -> None:
        self._tool_records.append(
            ToolRecord(
                name=tool_name,
                args_display=_format_args_display(args_json),
                result_preview=_format_result_preview(result),
                child_agent=child_agent,
            )
        )
