"""Optional local WikiBricks memory for Omnigent-managed sessions."""

from __future__ import annotations

import asyncio
import getpass
import importlib.util
import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from omnigent.tools.base import Tool, ToolContext

if TYPE_CHECKING:
    from omnigent.entities import ConversationItem

_logger = logging.getLogger(__name__)

WIKIBRICKS_TOOL_NAMES = (
    "wiki_search",
    "wiki_read_full",
    "wiki_index",
    "wiki_write_page",
    "wiki_promote_answer",
)

_MEMORY_START = "<wikibricks-memory>"
_MEMORY_END = "</wikibricks-memory>"
_MEMORY_PATTERN = re.compile(
    rf"(?:\n{{0,2}})?{re.escape(_MEMORY_START)}\n.*?\n{re.escape(_MEMORY_END)}",
    flags=re.IGNORECASE | re.DOTALL,
)
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def wikibricks_available() -> bool:
    """Return whether the optional local memory package can be used."""
    enabled = os.environ.get("WIKIBRICKS_ENABLED", "true").strip().lower()
    return enabled not in _FALSE_VALUES and importlib.util.find_spec("wikibricks") is not None


def _new_client() -> Any:
    from wikibricks import WikiClient

    return WikiClient()


def _safe_user_id(items: list[ConversationItem]) -> str:
    configured = os.environ.get("WIKIBRICKS_USER_ID")
    identity = configured or next(
        (item.created_by for item in items if item.created_by),
        None,
    )
    value = identity or getpass.getuser()
    return value.replace("@", "-at-").replace("/", "-").replace("\\", "-")


def _source_harness(conversation: Any) -> str | None:
    override = getattr(conversation, "harness_override", None)
    if isinstance(override, str) and override and override != "auto":
        return override
    labels = getattr(conversation, "labels", None)
    wrapper = labels.get("omnigent.wrapper") if isinstance(labels, dict) else None
    if wrapper == "claude-code-native-ui":
        return "claude-code"
    if isinstance(wrapper, str) and wrapper.endswith("-native-ui"):
        return wrapper.removesuffix("-native-ui")
    return None


def _text_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in strip_memory_context(content):
        if isinstance(block, dict):
            text = block.get("text") or block.get("output_text")
            if isinstance(text, str):
                parts.append(text)
        elif isinstance(block, str):
            parts.append(block)
    return "\n".join(parts).strip()


def _session_event(item: ConversationItem, *, source_harness: str | None, runner_id: str | None):
    from wikibricks.models import SessionEvent

    data = item.data.model_dump(exclude_none=True, by_alias=True)
    metadata: dict[str, Any] = {
        "response_id": item.response_id,
        "source_harness": source_harness,
        "runner_id": runner_id,
        "created_by": item.created_by,
    }
    metadata = {key: value for key, value in metadata.items() if value is not None}
    kind: str
    content: str
    if item.type == "message":
        if data.get("is_meta"):
            return None
        role = data.get("role")
        if role not in {"user", "assistant"}:
            return None
        kind = str(role)
        content = _text_content(data.get("content"))
        agent = data.get("model")
        if agent is not None:
            metadata["agent"] = agent
    elif item.type == "function_call":
        kind = "tool_call"
        content = str(data.get("arguments") or "")
        metadata.update(
            {
                "tool_name": data.get("name"),
                "call_id": data.get("call_id"),
            }
        )
    elif item.type == "function_call_output":
        kind = "tool_result"
        content = str(data.get("output") or "")
        metadata["call_id"] = data.get("call_id")
    elif item.type == "error":
        kind = "error"
        content = str(data.get("message") or data.get("code") or "error")
        metadata["code"] = data.get("code")
    else:
        return None
    if not content.strip():
        return None
    metadata = {key: value for key, value in metadata.items() if value is not None}
    created_at = datetime.fromtimestamp(item.created_at, tz=timezone.utc).isoformat()
    return SessionEvent(
        external_id=item.id,
        kind=kind,
        content=content,
        created_at=created_at,
        metadata=metadata,
    )


@dataclass
class _PendingCapture:
    source: Any
    items: OrderedDict[str, ConversationItem]


class WikiBricksMemory:
    """Bounded, deduplicating background bridge to local WikiBricks."""

    def __init__(
        self,
        *,
        client_factory: Callable[[], Any] | None = None,
        max_pending: int = 64,
    ) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self._client_factory = client_factory or _new_client
        self._enabled = client_factory is not None or wikibricks_available()
        self._client: Any | None = None
        self._client_lock = threading.Lock()
        self._condition = threading.Condition()
        self._pending: OrderedDict[str, _PendingCapture] = OrderedDict()
        self._active = 0
        self._closed = False
        self._max_pending = max_pending
        self._thread: threading.Thread | None = None
        self._automation_stop: threading.Event | None = None
        self._automation_thread: threading.Thread | None = None
        if self._enabled:
            self._thread = threading.Thread(
                target=self._run,
                name="omnigent-wikibricks",
                daemon=True,
            )
            self._thread.start()
            if client_factory is None:
                self._automation_stop = threading.Event()
                self._automation_thread = threading.Thread(
                    target=self._run_automation,
                    name="omnigent-wikibricks-maintenance",
                    daemon=True,
                )
                self._automation_thread.start()

    def _run_automation(self) -> None:
        from wikibricks.automation import run_background_worker

        assert self._automation_stop is not None
        run_background_worker(self._automation_stop)

    def _get_client(self) -> Any:
        with self._client_lock:
            if self._client is None:
                self._client = self._client_factory()
            return self._client

    def capture(
        self,
        source: Any,
        conversation_id: str,
        items: list[ConversationItem],
    ) -> bool:
        """Queue committed items without blocking the conversation writer."""
        if not self._enabled or not items:
            return False
        with self._condition:
            if self._closed:
                return False
            pending = self._pending.get(conversation_id)
            if pending is None:
                if len(self._pending) >= self._max_pending:
                    dropped_id, _ = self._pending.popitem(last=False)
                    _logger.warning(
                        "WikiBricks capture queue full; dropped pending session %s",
                        dropped_id,
                    )
                pending = _PendingCapture(source=source, items=OrderedDict())
                self._pending[conversation_id] = pending
            else:
                pending.source = source
            for item in items:
                pending.items[item.id] = item
            self._condition.notify()
        return True

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait()
                if self._closed and not self._pending:
                    return
                conversation_id, pending = self._pending.popitem(last=False)
                self._active += 1
            try:
                self._capture_now(conversation_id, pending)
            except Exception:  # noqa: BLE001 - optional memory must fail open
                _logger.warning(
                    "WikiBricks capture failed for session %s",
                    conversation_id,
                    exc_info=True,
                )
            finally:
                with self._condition:
                    self._active -= 1
                    self._condition.notify_all()

    def _capture_now(self, conversation_id: str, pending: _PendingCapture) -> None:
        from wikibricks.models import SessionRecord

        conversation = pending.source.get_conversation(conversation_id)
        if conversation is None or getattr(conversation, "archived", False):
            return
        if getattr(conversation, "kind", "default") != "default":
            return
        source_harness = _source_harness(conversation)
        runner_id = getattr(conversation, "runner_id", None)
        new_events = [
            event
            for item in pending.items.values()
            if (event := _session_event(
                item,
                source_harness=source_harness,
                runner_id=runner_id,
            ))
            is not None
        ]
        if not new_events:
            return
        client = self._get_client()
        previous = client.store.read_session_events("omnigent", conversation_id)
        merged = OrderedDict((event.external_id, event) for event in previous)
        for event in new_events:
            merged[event.external_id] = event
        metadata = {
            "title": getattr(conversation, "title", None),
            "source_harness": source_harness,
            "runner_id": runner_id,
        }
        client.ingest_session(
            SessionRecord(
                harness="omnigent",
                external_id=conversation_id,
                user_id=_safe_user_id(list(pending.items.values())),
                agent=getattr(conversation, "agent_id", None),
                workspace=getattr(conversation, "workspace", None),
                started_at=datetime.fromtimestamp(
                    conversation.created_at,
                    tz=timezone.utc,
                ).isoformat(),
                updated_at=datetime.fromtimestamp(
                    conversation.updated_at,
                    tz=timezone.utc,
                ).isoformat(),
                events=list(merged.values()),
                metadata={key: value for key, value in metadata.items() if value is not None},
            )
        )

    def retrieve(
        self,
        *,
        query: str,
        user_id: str,
        workspace: str | None,
        current_session_id: str,
        max_chars: int = 6_000,
    ) -> str:
        """Return bounded context for one turn, or an empty string when disabled."""
        if not self._enabled or not query.strip():
            return ""
        from wikibricks.models import MemoryQuery

        packet = self._get_client().retrieve_memory(
            MemoryQuery(
                text=query,
                user_id=user_id,
                workspace=workspace,
                current_session_id=current_session_id,
                max_chars=max_chars,
            )
        )
        return packet.rendered

    def flush(self, *, timeout: float = 5.0) -> bool:
        """Wait until queued captures finish. Intended for shutdown and tests."""
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._pending or self._active:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
        return True

    def close(self) -> None:
        """Stop the daemon after pending work has been flushed by the caller."""
        if self._automation_stop is not None:
            self._automation_stop.set()
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self._automation_thread is not None:
            self._automation_thread.join(timeout=1)


class _WikiBricksTool(Tool):
    def __init__(self, schema: dict[str, Any]) -> None:
        self._schema = schema

    def name(self) -> str:  # type: ignore[override]
        return str(self._schema["name"])

    def description(self) -> str:  # type: ignore[override]
        return str(self._schema.get("description") or "")

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name(),
                "description": self.description(),
                "parameters": self._schema.get(
                    "inputSchema",
                    {"type": "object", "properties": {}},
                ),
            },
        }

    def invoke(self, arguments: str, ctx: ToolContext) -> str:
        try:
            parsed = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"invalid JSON arguments: {exc.msg}"})
        if not isinstance(parsed, dict):
            return json.dumps({"error": "tool arguments must be a JSON object"})
        if self.name() in {"wiki_write_page", "wiki_promote_answer"}:
            parsed.setdefault("created_by", f"omnigent:{ctx.agent_id}")
        from wikibricks.mcp_server import format_tool_response

        return format_tool_response(self.name(), parsed)


def build_wikibricks_tools() -> list[Tool]:
    """Build the five local tools when WikiBricks is installed and enabled."""
    if not wikibricks_available():
        return []
    from wikibricks.resources import get_tool_schemas

    schemas = get_tool_schemas()
    names = tuple(schema.get("name") for schema in schemas)
    if names != WIKIBRICKS_TOOL_NAMES:
        raise RuntimeError(f"unexpected WikiBricks tool contract: {names}")
    return [_WikiBricksTool(schema) for schema in schemas]


def strip_memory_context(content: object) -> object:
    """Remove injected WikiBricks blocks before persistence or UI rendering."""
    if not isinstance(content, list):
        return content
    cleaned: list[object] = []
    for block in content:
        if not isinstance(block, dict):
            cleaned.append(block)
            continue
        text = block.get("text")
        if not isinstance(text, str):
            cleaned.append(block)
            continue
        stripped = _MEMORY_PATTERN.sub("", text)
        if not stripped:
            continue
        if stripped == text:
            cleaned.append(block)
        else:
            updated = dict(block)
            updated["text"] = stripped
            cleaned.append(updated)
    return cleaned


async def prepare_turn_memory(
    body: Any,
    *,
    conversation_id: str,
    user_id: str,
    workspace: str | None,
    memory: WikiBricksMemory | None = None,
    max_chars: int = 6_000,
    timeout: float = 0.25,
) -> Any:
    """Append relevant local context to one user event without exposing it in history."""
    if body.type != "message" or body.data.get("role") != "user":
        return body
    content = body.data.get("content")
    query = _text_content(content)
    if not query:
        return body
    active = memory or _default_memory()
    try:
        rendered = await asyncio.wait_for(
            asyncio.to_thread(
                active.retrieve,
                query=query,
                user_id=user_id,
                workspace=workspace,
                current_session_id=conversation_id,
                max_chars=max_chars,
            ),
            timeout=timeout,
        )
    except Exception:  # noqa: BLE001 - optional recall must fail open
        _logger.debug(
            "WikiBricks recall skipped for session %s",
            conversation_id,
            exc_info=True,
        )
        return body
    if not rendered:
        return body
    data = dict(body.data)
    clean_content = strip_memory_context(content)
    if not isinstance(clean_content, list):
        return body
    data["content"] = [
        *clean_content,
        {
            "type": "input_text",
            "text": f"{_MEMORY_START}\n{rendered}\n{_MEMORY_END}",
        },
    ]
    return body.model_copy(update={"data": data})


_default_lock = threading.Lock()
_default_instance: WikiBricksMemory | None = None


def _default_memory() -> WikiBricksMemory:
    global _default_instance
    with _default_lock:
        if _default_instance is None:
            _default_instance = WikiBricksMemory()
        return _default_instance


def capture_committed_items(
    source: Any,
    conversation_id: str,
    items: list[ConversationItem],
) -> None:
    """Best-effort notification called only after the Omnigent commit succeeds."""
    try:
        _default_memory().capture(source, conversation_id, items)
    except Exception:  # noqa: BLE001 - optional capture must fail open
        _logger.warning(
            "Could not queue WikiBricks capture for session %s",
            conversation_id,
            exc_info=True,
        )


__all__ = [
    "WIKIBRICKS_TOOL_NAMES",
    "WikiBricksMemory",
    "build_wikibricks_tools",
    "capture_committed_items",
    "prepare_turn_memory",
    "strip_memory_context",
    "wikibricks_available",
]
