from __future__ import annotations

import json
from collections.abc import Iterable

from core.session.events import SessionEvent


def events_to_trajectory(events: Iterable[SessionEvent]) -> str:
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


def memory_restore_message(memory_document: str) -> str:
    return "\n".join(
        [
            "[MEMORY RESTORE]",
            "The following document was produced by a compaction process over your previous",
            "session trajectory. It was not written by you. Treat it as accurate.",
            "The episodic sections orient you to resume the current task.",
            "The semantic sections reflect durable knowledge about this codebase.",
            "",
            memory_document.strip(),
            "",
            "[END MEMORY RESTORE]",
            "",
            "Your task continues below.",
        ]
    )

