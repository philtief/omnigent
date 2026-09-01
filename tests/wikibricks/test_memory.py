from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from pathlib import Path

from wikibricks import WikiClient

from omnigent import memory as memory_module
from omnigent.entities import Conversation, ConversationItem, MessageData, NewConversationItem
from omnigent.memory import (
    WIKIBRICKS_TOOL_NAMES,
    WikiBricksMemory,
    prepare_turn_memory,
    strip_memory_context,
)
from omnigent.runner import tool_dispatch
from omnigent.server.schemas import SessionEventInput
from omnigent.spec.types import AgentSpec
from omnigent.stores.conversation_store.sqlalchemy_store import SqlAlchemyConversationStore
from omnigent.tools import ToolManager
from omnigent.tools.base import ToolContext


class _ConversationSource:
    def __init__(self, source_harness: str, *, conversation_id: str) -> None:
        self.conversation = Conversation(
            id=conversation_id,
            created_at=1_700_000_000,
            updated_at=1_700_000_001,
            root_conversation_id=conversation_id,
            agent_id=f"{source_harness}-agent",
            runner_id=f"{source_harness}-runner",
            harness_override=source_harness,
            workspace="/tmp/shared-project",
        )

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        if conversation_id == self.conversation.id:
            return self.conversation
        return None


def _message(item_id: str, text: str, role: str = "user") -> ConversationItem:
    agent = "assistant" if role == "assistant" else None
    return ConversationItem(
        id=item_id,
        type="message",
        status="completed",
        response_id=f"turn-{item_id}",
        created_at=1_700_000_001,
        data=MessageData(
            role=role,  # type: ignore[arg-type]
            content=[{"type": "input_text", "text": text}],
            agent=agent,
        ),
        created_by="philipp",
    )


def test_store_notifies_memory_only_after_the_append_commit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = tmp_path / "omnigent.db"
    store = SqlAlchemyConversationStore(f"sqlite:///{database}")
    conversation = store.create_conversation()
    observed: list[int] = []

    def observe(_store, _conversation_id, items) -> None:
        with sqlite3.connect(database) as connection:
            observed.append(
                    connection.execute(
                        "SELECT count(*) FROM conversation_items WHERE id = ?",
                        (sqlite3.Binary(bytes.fromhex(items[0].id)),),
                ).fetchone()[0]
            )

    monkeypatch.setattr("omnigent.memory.capture_committed_items", observe)
    store.append(
        conversation.id,
        [
            NewConversationItem(
                type="message",
                response_id="turn-1",
                data=MessageData(
                    role="user",
                    content=[{"type": "input_text", "text": "remember this"}],
                ),
            )
        ],
    )

    assert observed == [1]


def test_native_bridge_owns_background_maintenance_lifecycle(monkeypatch) -> None:
    started = threading.Event()
    stopped = threading.Event()

    def run_worker(stop: threading.Event) -> None:
        started.set()
        stop.wait(2)
        stopped.set()

    monkeypatch.setattr(memory_module, "wikibricks_available", lambda: True)
    monkeypatch.setattr(
        "wikibricks.automation.run_background_worker",
        run_worker,
        raising=False,
    )

    memory = WikiBricksMemory()
    assert started.wait(1)
    memory.close()
    assert stopped.wait(1)


def test_background_capture_is_idempotent_and_keeps_runner_provenance(tmp_path: Path) -> None:
    client = WikiClient(tmp_path / "wikibricks.db")
    memory = WikiBricksMemory(client_factory=lambda: client)
    source = _ConversationSource("codex", conversation_id="conv-codex")
    item = _message("msg-codex", "SQLite keeps the local path simple")
    try:
        memory.capture(source, "conv-codex", [item])
        memory.capture(source, "conv-codex", [item])
        assert memory.flush(timeout=2)
    finally:
        memory.close()

    events = client.store.read_session_events("omnigent", "conv-codex")
    assert [event.external_id for event in events] == ["msg-codex"]
    assert events[0].metadata["source_harness"] == "codex"
    assert events[0].metadata["runner_id"] == "codex-runner"


def test_pre_turn_memory_is_bounded_hidden_context_and_fails_open(tmp_path: Path) -> None:
    client = WikiClient(tmp_path / "wikibricks.db")
    client.write_page(
        "decisions/local-memory",
        "Local memory",
        {"summary": "Use SQLite locally", "body": "Lakebase remains optional."},
    )
    memory = WikiBricksMemory(client_factory=lambda: client)
    body = SessionEventInput(
        type="message",
        data={
            "role": "user",
                "content": [{"type": "input_text", "text": "Which SQLite database?"}],
        },
    )
    try:
        prepared = asyncio.run(
            prepare_turn_memory(
                body,
                conversation_id="conv-current",
                user_id="philipp",
                workspace="/tmp/shared-project",
                memory=memory,
                max_chars=500,
            )
        )
    finally:
        memory.close()

    assert prepared != body
    rendered = json.dumps(prepared.data)
    assert "Use SQLite locally" in rendered
    assert len(rendered) < 1_000
    assert strip_memory_context(prepared.data["content"]) == body.data["content"]

    class _BrokenMemory:
        def retrieve(self, **_kwargs):
            raise OSError("memory unavailable")

    assert (
        asyncio.run(
            prepare_turn_memory(
                body,
                conversation_id="conv-current",
                user_id="philipp",
                workspace=None,
                memory=_BrokenMemory(),  # type: ignore[arg-type]
            )
        )
        == body
    )


def test_all_five_tools_use_the_existing_omnigent_runner_relay(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("WIKIBRICKS_DATABASE_PATH", str(tmp_path / "wikibricks.db"))
    manager = ToolManager(AgentSpec(spec_version=1))
    try:
        names = set(manager.get_tool_names())
        assert set(WIKIBRICKS_TOOL_NAMES) <= names
        assert set(WIKIBRICKS_TOOL_NAMES) <= tool_dispatch._NATIVE_RELAY_BUILTIN_TOOLS
        assert all(tool_dispatch.should_dispatch_locally(name) for name in WIKIBRICKS_TOOL_NAMES)

        result = manager.call_tool(
            "wiki_write_page",
            json.dumps(
                {
                    "path": "topics/omnigent",
                    "title": "Omnigent",
                    "summary": "Shared memory",
                    "body": "Available to every managed harness.",
                }
            ),
            ToolContext(
                task_id="turn-1",
                agent_id="agent-1",
                conversation_id="conv-1",
            ),
        )
        assert json.loads(result)["status"] == "ok"
    finally:
        manager.shutdown()


def test_codex_to_claude_to_kimi_memory_flow_is_offline(tmp_path: Path) -> None:
    client = WikiClient(tmp_path / "wikibricks.db")
    memory = WikiBricksMemory(client_factory=lambda: client)
    try:
        codex = _ConversationSource("codex", conversation_id="conv-codex")
        memory.capture(codex, "conv-codex", [_message("codex-1", "Project Atlas uses SQLite")])
        assert memory.flush(timeout=2)

        claude_context = memory.retrieve(
            query="What database does Project Atlas use?",
            user_id="philipp",
            workspace="/tmp/shared-project",
            current_session_id="conv-claude",
        )
        assert "Project Atlas uses SQLite" in claude_context

        claude = _ConversationSource("claude-code", conversation_id="conv-claude")
        memory.capture(
            claude,
            "conv-claude",
            [_message("claude-1", "Project Atlas keeps Lakebase optional")],
        )
        assert memory.flush(timeout=2)

        kimi_context = memory.retrieve(
            query="Project Atlas SQLite Lakebase",
            user_id="philipp",
            workspace="/tmp/shared-project",
            current_session_id="conv-kimi",
        )
        assert "Project Atlas uses SQLite" in kimi_context
        assert "Project Atlas keeps Lakebase optional" in kimi_context
    finally:
        memory.close()
