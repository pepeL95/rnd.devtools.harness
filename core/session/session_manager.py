from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from langchain_core.messages import BaseMessage

from core.utilities.messages import (
    make_message,
    message_reasoning_blocks,
    message_role,
    message_text_content,
    message_tool_calls,
)
from core.session.events import EventType, RuntimeSnapshot, SessionEvent
from core.session.io import append_events, read_events, replace_events, session_paths
from core.session.turns import agent_history_events, next_turn


class SessionManager:
    """Owns session persistence in dump and curated JSONL streams."""

    def __init__(self, session_id: str | None = None, root: Path | None = None) -> None:
        self.session_id = session_id or uuid4().hex
        self.dump_path, self.curated_path = session_paths(self.session_id, root)

    def read_dump(self) -> list[SessionEvent]:
        return read_events(self.dump_path)

    def read_curated(self) -> list[SessionEvent]:
        return read_events(self.curated_path)

    def append(self, events: Iterable[SessionEvent], curated: bool = True) -> None:
        materialized = list(events)
        if not materialized:
            return
        append_events(self.dump_path, materialized)
        if curated:
            append_events(self.curated_path, materialized)

    def replace_curated(self, events: Iterable[SessionEvent]) -> None:
        replace_events(self.curated_path, events)

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
        for event in agent_history_events(self.read_curated()):
            role = str(event.payload.get("role") or event.type.value)
            content = event.payload.get("content", "")
            restored.append(make_message(role=role, content=content))
        return restored
