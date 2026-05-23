from __future__ import annotations

from collections.abc import Iterable

from core.session.events import EventType, SessionEvent


def next_turn(events: Iterable[SessionEvent]) -> int:
    highest = max((event.turn for event in events), default=0)
    return highest + 1


def conversational_events(events: Iterable[SessionEvent]) -> list[SessionEvent]:
    allowed = {EventType.USER, EventType.ASSISTANT, EventType.SYSTEM, EventType.TOOL, EventType.TOOL_OUTPUT}
    return [event for event in events if event.type in allowed]


def agent_history_events(events: Iterable[SessionEvent]) -> list[SessionEvent]:
    """Events safe to restore into a LangChain agent transcript."""
    allowed = {EventType.USER, EventType.ASSISTANT}
    return [event for event in events if event.type in allowed]

