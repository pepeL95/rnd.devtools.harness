from __future__ import annotations

from collections.abc import Iterable

from core.session.events import EventType, SessionEvent

HIDDEN_HISTORY_KINDS = {"memory_restore", "trajectory_memory"}


def next_turn(events: Iterable[SessionEvent]) -> int:
    highest = max((event.turn for event in events), default=0)
    return highest + 1


def agent_history_events(events: Iterable[SessionEvent]) -> list[SessionEvent]:
    """Events safe to restore into a LangChain agent transcript."""
    allowed = {EventType.USER, EventType.ASSISTANT}
    return [event for event in events if event.type in allowed]


def display_history_events(events: Iterable[SessionEvent]) -> list[SessionEvent]:
    """User/assistant history suitable for the chat UI from the full dump stream."""
    return [
        event
        for event in events
        if event.type in {EventType.USER, EventType.ASSISTANT}
        and event.payload.get("kind") not in HIDDEN_HISTORY_KINDS
    ]
