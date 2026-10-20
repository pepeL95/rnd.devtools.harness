from __future__ import annotations

import json
from collections.abc import Iterable

from core.compaction.models import TaskArc
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
            "The task map preserves prior work, decisions, and constraints.",
            "Use the current working state and retained turns to continue without repeating work.",
            "",
            memory_document.strip(),
            "",
            "[END MEMORY RESTORE]",
            "",
            "Your task continues below.",
        ]
    )


def render_arc_memory(
    narrative: str, current: str, arcs: list[TaskArc], session: str, manifest: dict[str, list[str]],
) -> str:
    parts = ["# Session Arc", "## Direction", narrative, "## Task Map"]
    for arc in arcs:
        parts.append(f"### {arc.title} [{arc.status}]")
        for label, value in (("Intent", arc.intent), ("Implementation", arc.implementation), ("Findings", arc.findings),
                             ("Constraints", arc.constraints), ("Validation", arc.validation), ("Remaining", arc.remaining)):
            parts.append(f"{label}: {value}")
        if manifest.get(arc.arc_id):
            reference = (f"trajectory://{session}/{arc.arc_id}" if session else
                         ", ".join(f"trajectory-record://{id}" for id in manifest[arc.arc_id]))
            parts.append(f"Evidence: {reference} ({len(manifest[arc.arc_id])} trajectories)")
    parts.extend(["## Current Working State", current])
    return "\n\n".join(parts)

