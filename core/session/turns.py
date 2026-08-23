from __future__ import annotations

from collections.abc import Iterable

from core.session.events import EventType, SessionEvent

HARNESS_CONTEXT_KIND = "harness_context"
NON_USER_AUTHORED_KINDS = {"memory_restore", "trajectory_memory", HARNESS_CONTEXT_KIND}
HIDDEN_HISTORY_KINDS = NON_USER_AUTHORED_KINDS


def next_turn(events: Iterable[SessionEvent]) -> int:
    highest = max((event.turn for event in events), default=0)
    return highest + 1


def agent_history_events(events: Iterable[SessionEvent]) -> list[SessionEvent]:
    """Events safe to restore into a LangChain agent transcript.

    Includes the full internal trajectory (tool calls, tool outputs, reasoning
    blocks) so the agent re-enters cross-turn with complete context rather than
    only the surface-level user/assistant dialogue.
    """
    allowed = {
        EventType.USER,
        EventType.ASSISTANT,
        EventType.TOOL,
        EventType.TOOL_OUTPUT,
        EventType.REASONING,
    }
    return [event for event in events if event.type in allowed]


def is_user_authored_event(event: SessionEvent) -> bool:
    return event.type == EventType.USER and event.payload.get("kind") not in NON_USER_AUTHORED_KINDS


def display_history_events(events: Iterable[SessionEvent]) -> list[SessionEvent]:
    """User/assistant history suitable for the chat UI from the full dump stream."""
    return [
        event
        for event in events
        if event.type in {EventType.USER, EventType.ASSISTANT}
        and event.payload.get("kind") not in HIDDEN_HISTORY_KINDS
    ]
