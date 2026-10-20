from __future__ import annotations

import json
from datetime import datetime, timezone

from core.session.events import SessionEvent, is_pinned_instruction
from core.session.manager import SessionManager
from core.trajectory_bank.evidence import content_hash, fingerprint, keywords as extract_keywords
from core.trajectory_bank.models import TrajectoryRecord


def prepare_trajectories(
    manager: SessionManager,
    new_events: list[SessionEvent],
    turns: list[int],
    episodes: list[str],
) -> list[TrajectoryRecord]:
    """Build immutable task episodes with conservative eviction-batch provenance."""
    raw_events = manager.read_dump()
    source = [event for event in raw_events if event.turn in turns and not is_pinned_instruction(event)]
    if {event.turn for event in source} != set(turns):
        raise ValueError("Raw turn evidence is missing; refusing to archive unverifiable trajectories")
    records = []
    timestamp = datetime.now(timezone.utc).isoformat()
    group = "batch-" + fingerprint(new_events)[:20]
    source_hash = fingerprint(source)
    for index, document in enumerate(episodes):
        digest = content_hash(document)
        record_id = f"{manager.session_id}-{group}-e{index}-{source_hash[:12]}-{digest[:12]}"
        metadata = {
            "schema_version": 1, "session_id": manager.session_id, "arc_id": group, "turn": turns[0],
            "source_turns": json.dumps(turns), "episode_index": index,
            "started_at": source[0].timestamp, "ended_at": source[-1].timestamp, "archived_at": timestamp,
            "source_event_ids": json.dumps([event.id for event in source]), "source_path": str(manager.dump_path),
            "source_fingerprint": source_hash, "content_hash": digest,
            "keywords": json.dumps(extract_keywords(document)), "source_kind": "task_episode",
        }
        records.append(TrajectoryRecord(record_id, document, metadata))

    return records
