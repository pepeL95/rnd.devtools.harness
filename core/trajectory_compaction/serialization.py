from __future__ import annotations

import json
from collections.abc import Iterable

from core.session.events import SessionEvent
from core.trajectory_compaction.models import TurnTrajectorySynthesis


def events_to_internal_trajectory(events: Iterable[SessionEvent]) -> str:
    lines: list[str] = []
    for index, event in enumerate(events, start=1):
        payload = json.dumps(event.payload, ensure_ascii=False, sort_keys=True)
        lines.append(
            "\n".join(
                [
                    f"EVENT {index}",
                    f"timestamp: {event.timestamp}",
                    f"type: {event.type.value}",
                    f"turn: {event.turn}",
                    f"payload: {payload}",
                ]
            )
        )
    return "\n\n".join(lines)


def trajectory_memory_message(turn_synthesis: TurnTrajectorySynthesis) -> str:
    return "\n".join(
        [
            "[TRAJECTORY MEMORY]",
            "",
            turn_synthesis.synthesis.strip(),
            "",
            "Live edge:",
            turn_synthesis.live_edge.strip(),
            "",
            "[END TRAJECTORY MEMORY]",
        ]
    )


def format_turn_interval(turns: list[int]) -> str:
    if not turns:
        return "[]"
    ordered = sorted(set(turns))
    ranges: list[str] = []
    start = ordered[0]
    previous = ordered[0]
    for turn in ordered[1:]:
        if turn == previous + 1:
            previous = turn
            continue
        ranges.append(_range_label(start, previous))
        start = previous = turn
    ranges.append(_range_label(start, previous))
    return "[" + ", ".join(ranges) + "]"


def _range_label(start: int, end: int) -> str:
    if start == end:
        return str(start)
    return f"{start}-{end}"
