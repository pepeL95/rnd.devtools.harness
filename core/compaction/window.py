from __future__ import annotations

from core.compaction.models import CompactionWindow
from core.session.events import SessionEvent


def split_compaction_window(events: list[SessionEvent], keep_last_turns: int) -> CompactionWindow:
    if not events:
        return CompactionWindow(compacted=[], retained=[])
    if keep_last_turns == 0:
        return CompactionWindow(compacted=list(events), retained=[])

    turns = sorted({event.turn for event in events})
    retained_turns = set(turns[-keep_last_turns:])
    compacted = [event for event in events if event.turn not in retained_turns]
    retained = [event for event in events if event.turn in retained_turns]
    return CompactionWindow(compacted=compacted, retained=retained)

