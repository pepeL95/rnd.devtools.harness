import json

import pytest

from core.session.events import EventType, SessionEvent
from core.session.manager import SessionManager, StaleCompactionError
from core.telemetry.store import TelemetryStore
from core.trajectory.compactor import TrajectoryCompactor
from core.trajectory.coordinator import PendingTrajectoryCompaction, TrajectoryCompactionCoordinator


def test_compaction_cannot_discard_a_live_correction(tmp_path):
    manager = SessionManager(root=tmp_path / "sessions")
    source = SessionEvent(type=EventType.USER, turn=1, payload={"content": "Task"})
    manager.append([source])
    correction = SessionEvent(type=EventType.USER, turn=1, payload={"content": "Keep exact tabs"})
    manager.append([correction])
    with pytest.raises(StaleCompactionError):
        manager.apply_compaction_result([], snapshot_latest_turn=1, source_event_ids=[source.id])
    assert manager.read_curated() == [source, correction]

    manager.sync_system_instruction("Latest instructions", kind="agents_md", source="AGENTS.md", turn=1)
    newer = SessionEvent(type=EventType.USER, turn=2, payload={"content": "Continue"})
    manager.append([newer])
    manager.apply_compaction_result([source, correction], snapshot_latest_turn=1,
                                    source_event_ids=[source.id, correction.id])
    assert manager.read_curated()[0].payload["content"] == "Latest instructions"
    assert manager.read_curated()[-1] == newer


def test_pending_trajectory_cannot_overwrite_a_new_session_arc_after_restart(tmp_path):
    manager = SessionManager(root=tmp_path / "sessions")
    source = SessionEvent(type=EventType.USER, turn=1, payload={"content": "Task"})
    manager.append([source])
    telemetry = TelemetryStore(tmp_path / "telemetry.jsonl")
    coordinator = TrajectoryCompactionCoordinator(manager, TrajectoryCompactor(), repo_root=tmp_path,
                                                 telemetry_store=telemetry)
    pending = PendingTrajectoryCompaction(result_events=[source], snapshot_latest_turn=1,
        payload={"counts": {}, "token_usage": {}}, compacted_event_count=1, compacted_turn_count=1,
        result_event_count=1, model_names={}, token_usage={}, duration_ms=1, source_event_ids=[source.id])
    coordinator._store_pending(pending)
    arc = SessionEvent(type=EventType.USER, turn=1, payload={"kind": "memory_restore", "content": "New arc"})
    manager.replace_curated([arc])
    restarted = TrajectoryCompactionCoordinator(manager, TrajectoryCompactor(), repo_root=tmp_path,
                                               telemetry_store=telemetry)
    assert restarted.prepare_for_agent() == "stale"
    assert restarted.prepare_for_agent() == "not_ready"
    assert manager.read_curated() == [arc]
    assert json.loads(telemetry.path.read_text().splitlines()[-1])["payload"]["reason"] == "stale_snapshot"
