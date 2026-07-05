from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from core.utilities.messages import (
    make_message,
    message_reasoning_blocks,
    message_role,
    message_text_content,
    message_tool_calls,
    normalize_message_content,
)
from core.session.events import EventType, RuntimeSnapshot, SessionEvent
from core.session.io import (
    append_events,
    read_events,
    replace_events,
    session_paths,
)
from core.session.turns import agent_history_events, display_history_events, next_turn


class SessionManager:
    """Owns session persistence in dump and curated JSONL streams."""

    def __init__(self, session_id: str | None = None, root: Path | None = None) -> None:
        self.session_id = session_id or uuid4().hex
        self.root = root
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

    def apply_compaction_result(
        self,
        compacted_events: Iterable[SessionEvent],
        *,
        snapshot_latest_turn: int,
    ) -> list[SessionEvent]:
        current = self.read_curated()
        preserved = [event for event in current if event.turn > snapshot_latest_turn]
        merged = [*list(compacted_events), *preserved]
        replace_events(self.curated_path, merged)
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
                "python_interpreter": snapshot.python_interpreter,
            },
        )
        self.append([event])
        return event

    def latest_runtime_snapshot(self) -> RuntimeSnapshot | None:
        for event in reversed(self.read_dump()):
            if event.type != EventType.RUNTIME:
                continue
            payload = event.payload
            cwd = payload.get("cwd")
            if not cwd:
                continue
            return RuntimeSnapshot(
                cwd=str(cwd),
                git_branch=_optional_str(payload.get("git_branch")),
                git_dirty=_optional_bool(payload.get("git_dirty")),
                python_interpreter=_optional_str(payload.get("python_interpreter")),
            )
        return None

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
        return _reconstruct_agent_messages(agent_history_events(self.read_curated()))

    def read_display_history(self) -> list[SessionEvent]:
        return display_history_events(self.read_dump())


def _reconstruct_agent_messages(events: list[SessionEvent]) -> list[BaseMessage]:
    """Reconstruct a valid LangChain message sequence from stored session events.

    LangChain requires strict structural pairing: every AIMessage with tool_calls
    must be immediately followed by one ToolMessage per call with a matching
    tool_call_id. This function groups TOOL events into a single AIMessage,
    inlines REASONING blocks as thinking content, and maps TOOL_OUTPUT events
    to ToolMessages with correct IDs.
    """
    messages: list[BaseMessage] = []

    # Pending tool-call batch: accumulates TOOL + REASONING events that belong
    # to the same logical AIMessage before it is flushed.
    pending_tool_calls: list[dict[str, Any]] = []
    pending_thinking: list[dict[str, Any]] = []

    def flush_pending_tool_calls() -> None:
        if not pending_tool_calls:
            pending_thinking.clear()
            return
        content: list[dict[str, Any]] = [
            {"type": "thinking", "thinking": block["thinking"]}
            for block in pending_thinking
        ]
        messages.append(
            AIMessage(
                content=content or "",
                tool_calls=[
                    {"name": tc["name"], "args": tc["args"], "id": tc["id"]}
                    for tc in pending_tool_calls
                ],
            )
        )
        pending_tool_calls.clear()
        pending_thinking.clear()

    for event in events:
        if event.type == EventType.USER:
            flush_pending_tool_calls()
            content = normalize_message_content(event.payload.get("content", ""))
            messages.append(
                HumanMessage(
                    content=content,
                    additional_kwargs=_message_additional_kwargs(event),
                )
            )

        elif event.type == EventType.TOOL:
            # Accumulate into the pending batch — multiple TOOL events map to
            # multiple tool_calls on the same AIMessage.
            payload = event.payload
            pending_tool_calls.append(
                {
                    "name": str(payload.get("name") or "tool"),
                    "args": payload.get("args") or {},
                    "id": str(payload.get("tool_call_id") or ""),
                }
            )

        elif event.type == EventType.REASONING:
            # Skip middleware-injected steering introspections — they are session
            # metadata, not model-generated thinking blocks.
            if event.payload.get("reasoning_format") == "live_steering":
                continue
            text = str(event.payload.get("content") or "").strip()
            if not text:
                continue
            if pending_tool_calls:
                # Attach to the pending tool-call AIMessage.
                pending_thinking.append({"thinking": text})
            else:
                # Standalone reasoning block: emit as an AIMessage with thinking content.
                flush_pending_tool_calls()
                messages.append(AIMessage(content=[{"type": "thinking", "thinking": text}]))

        elif event.type == EventType.TOOL_OUTPUT:
            # A tool result always terminates the pending tool-call batch.
            flush_pending_tool_calls()
            content = normalize_message_content(event.payload.get("content", ""))
            tool_call_id = str(event.payload.get("tool_call_id") or "restored")
            messages.append(ToolMessage(content=str(content), tool_call_id=tool_call_id))

        elif event.type == EventType.ASSISTANT:
            flush_pending_tool_calls()
            content = normalize_message_content(event.payload.get("content", ""))
            messages.append(
                AIMessage(
                    content=content,
                    additional_kwargs=_message_additional_kwargs(event),
                )
            )

    flush_pending_tool_calls()
    return messages


def _message_additional_kwargs(event: SessionEvent) -> dict[str, Any]:
    kind = event.payload.get("kind")
    if not kind:
        return {}
    return {"session_kind": kind}


def _optional_str(value: Any) -> str | None:
    return str(value) if value is not None else None


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None
