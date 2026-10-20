from __future__ import annotations

import json
from datetime import datetime, timezone

from core.session.events import SessionEvent, is_pinned_instruction
from core.session.manager import SessionManager
from core.trajectory_bank.evidence import (
    content_hash, fingerprint, keywords as extract_keywords, reasoning_chain, trajectory_document,
)
from core.trajectory_bank.models import TrajectoryRecord


def prepare_trajectories(
    manager: SessionManager,
    new_events: list[SessionEvent],
    turns: list[int],
) -> list[TrajectoryRecord]:
    """Build evidence records for the eviction batch without writing to the bank."""
    raw_events = manager.read_dump()
    sources = {turn: [event for event in raw_events if event.turn == turn and not is_pinned_instruction(event)] for turn in turns}
    if any(not source for source in sources.values()):
        raise ValueError("Raw turn evidence is missing; refusing to archive unverifiable trajectories")
    records = []
    timestamp = datetime.now(timezone.utc).isoformat()
    group = "batch-" + fingerprint(new_events)[:20]
    for turn in turns:
        source = sources[turn]
        events_in_turn = [event for event in new_events if event.turn == turn]
        syntheses = [str(event.payload.get("content", "")) for event in events_in_turn
                     if event.payload.get("kind") == "trajectory_memory"]
        document = "\n\n".join(syntheses) if syntheses else reasoning_chain(events_in_turn)
        if not document.strip():
            # Do not fabricate reasoning evidence when a turn has none.
            continue
        document = trajectory_document(events_in_turn, document)
        turn_keywords = extract_keywords(document)
        if turn_keywords:
            document += "\n\nKeywords: " + ", ".join(turn_keywords)
        digest = content_hash(document)
        source_hash = fingerprint(source)
        record_id = f"{manager.session_id}-{group}-t{turn}-{source_hash[:12]}-{digest[:12]}"
        metadata = {
            "schema_version": 1, "session_id": manager.session_id, "arc_id": group, "turn": turn,
            "started_at": source[0].timestamp, "ended_at": source[-1].timestamp, "archived_at": timestamp,
            "source_event_ids": json.dumps([event.id for event in source]), "source_path": str(manager.dump_path),
            "source_fingerprint": source_hash, "content_hash": digest,
            "keywords": json.dumps(turn_keywords), "source_kind": "trajectory_synthesis" if syntheses else "reasoning_chain",
        }
        records.append(TrajectoryRecord(record_id, document, metadata))

    return records
