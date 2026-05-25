from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4
from dataclasses import dataclass

from langchain_core.messages import BaseMessage

from core.utilities.messages import (
    make_message,
    message_reasoning_blocks,
    message_role,
    message_text_content,
    message_tool_calls,
)
from core.session.events import EventType, RuntimeSnapshot, SessionEvent
from core.session.io import (
    append_events,
    compaction_paths,
    read_events,
    read_json,
    replace_events,
    session_paths,
    write_json_exclusive,
)
from core.session.turns import agent_history_events, display_history_events, next_turn


@dataclass(frozen=True)
class CuratedCompactionLease:
    token: str
    trigger: str
    snapshot_events: list[SessionEvent]


class SessionManager:
    """Owns session persistence in dump and curated JSONL streams."""

    def __init__(self, session_id: str | None = None, root: Path | None = None) -> None:
        self.session_id = session_id or uuid4().hex
        self.root = root
        self.dump_path, self.curated_path = session_paths(self.session_id, root)
        self.compaction_lock_path, self.compaction_pending_path = compaction_paths(self.session_id, root)

    def read_dump(self) -> list[SessionEvent]:
        return read_events(self.dump_path)

    def read_curated(self, include_pending: bool = True) -> list[SessionEvent]:
        curated = read_events(self.curated_path)
        if not include_pending:
            return curated
        return [*curated, *self.read_pending_curated()]

    def read_pending_curated(self) -> list[SessionEvent]:
        return read_events(self.compaction_pending_path)

    def append(self, events: Iterable[SessionEvent], curated: bool = True) -> None:
        materialized = list(events)
        if not materialized:
            return
        append_events(self.dump_path, materialized)
        if curated:
            target = self.compaction_pending_path if self.is_curated_locked() else self.curated_path
            append_events(target, materialized)

    def replace_curated(self, events: Iterable[SessionEvent]) -> None:
        replace_events(self.curated_path, events)

    def is_curated_locked(self) -> bool:
        return self.compaction_lock_path.exists()

    def current_compaction_lock(self) -> dict[str, Any] | None:
        return read_json(self.compaction_lock_path)

    def begin_curated_compaction(
        self,
        trigger: str,
        metadata: dict[str, Any] | None = None,
    ) -> CuratedCompactionLease | None:
        token = uuid4().hex
        payload = {"token": token, "trigger": trigger, **(metadata or {})}
        created = write_json_exclusive(self.compaction_lock_path, payload)
        if not created:
            return None
        return CuratedCompactionLease(
            token=token,
            trigger=trigger,
            snapshot_events=self.read_curated(include_pending=False),
        )

    def finalize_curated_compaction(
        self,
        lease: CuratedCompactionLease,
        compacted_events: Iterable[SessionEvent],
    ) -> list[SessionEvent]:
        self._assert_active_lease(lease)
        merged = [*list(compacted_events), *self.read_pending_curated()]
        replace_events(self.curated_path, merged)
        self._clear_compaction_artifacts()
        return merged

    def abort_curated_compaction(self, lease: CuratedCompactionLease) -> list[SessionEvent]:
        self._assert_active_lease(lease)
        merged = self.read_curated(include_pending=True)
        replace_events(self.curated_path, merged)
        self._clear_compaction_artifacts()
        return merged

    def next_turn(self) -> int:
        return next_turn(self.read_dump())

    def record_runtime(self, snapshot: RuntimeSnapshot, turn: int | None = None) -> SessionEvent:
        event = SessionEvent(
            type=EventType.RUNTIME,
            turn=turn or self.next_turn(),
            payload={
                "cwd": snapshot.cwd,
                "git_branch": snapshot.git_branch,
                "git_dirty": snapshot.git_dirty,
            },
        )
        self.append([event])
        return event

    def events_from_messages(self, messages: Iterable[Any], turn: int | None = None) -> list[SessionEvent]:
        turn_number = turn or self.next_turn()
        events: list[SessionEvent] = []
        for message in messages:
            role = message_role(message)
            message_type = getattr(message, "type", None)
            if role == "assistant":
                for index, call in enumerate(message_tool_calls(message)):
                    events.append(
                        SessionEvent(
                            type=EventType.TOOL,
                            turn=turn_number,
                            payload={
                                "role": role,
                                "name": str(call.get("name") or call.get("type") or "tool"),
                                "args": call.get("args") or call.get("arguments") or {},
                                "tool_call_id": call.get("id"),
                                "message_type": message_type,
                                "index": index,
                            },
                        )
                    )
                for index, block in enumerate(message_reasoning_blocks(message)):
                    events.append(
                        SessionEvent(
                            type=EventType.REASONING,
                            turn=turn_number,
                            payload={
                                "role": role,
                                "content": block["text"],
                                "message_type": message_type,
                                "reasoning_format": block["format"],
                                "signature": block["signature"],
                                "index": index,
                            },
                        )
                    )
                assistant_text = message_text_content(message)
                if assistant_text:
                    events.append(
                        SessionEvent(
                            type=EventType.ASSISTANT,
                            turn=turn_number,
                            payload={
                                "role": role,
                                "content": assistant_text,
                                "message_type": message_type,
                            },
                        )
                    )
                continue
            event_type = {
                "user": EventType.USER,
                "assistant": EventType.ASSISTANT,
                "system": EventType.SYSTEM,
                "tool": EventType.TOOL_OUTPUT,
            }.get(role, EventType.META)
            events.append(
                SessionEvent(
                    type=event_type,
                    turn=turn_number,
                    payload={
                        "role": role,
                        "content": getattr(message, "content", message),
                        "message_type": message_type,
                    },
                )
            )
        return events

    def load_curated_messages(self) -> list[BaseMessage]:
        restored: list[BaseMessage] = []
        for event in agent_history_events(self.read_curated(include_pending=True)):
            role = str(event.payload.get("role") or event.type.value)
            content = event.payload.get("content", "")
            restored.append(
                make_message(
                    role=role,
                    content=content,
                    additional_kwargs=_message_additional_kwargs(event),
                )
            )
        return restored

    def read_display_history(self) -> list[SessionEvent]:
        return display_history_events(self.read_dump())

    def _assert_active_lease(self, lease: CuratedCompactionLease) -> None:
        lock = self.current_compaction_lock()
        if lock is None or lock.get("token") != lease.token:
            raise RuntimeError("Curated compaction lease is no longer active.")

    def _clear_compaction_artifacts(self) -> None:
        if self.compaction_pending_path.exists():
            self.compaction_pending_path.unlink()
        if self.compaction_lock_path.exists():
            self.compaction_lock_path.unlink()


def _message_additional_kwargs(event: SessionEvent) -> dict[str, Any]:
    kind = event.payload.get("kind")
    if not kind:
        return {}
    return {"session_kind": kind}
