from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from threading import Lock, Thread
from time import perf_counter

from core.compaction.telemetry import CompactionTelemetry, compaction_logger, compaction_time_span
from core.compaction.token_counter import TokenCounter
from core.session.events import SessionEvent
from core.session.manager import SessionManager
from core.telemetry.store import TelemetryStore, telemetry_session_path
from core.trajectory.compactor import TrajectoryCompactor
from core.utilities.defaults import get_model_name


@dataclass(frozen=True)
class PendingTrajectoryCompaction:
    result_events: list[SessionEvent]
    snapshot_latest_turn: int
    payload: dict[str, object]
    compacted_event_count: int
    compacted_turn_count: int
    result_event_count: int
    model_names: dict[str, str]
    token_usage: dict[str, int]
    duration_ms: int


class TrajectoryCompactionCoordinator:
    """Run trajectory compaction in the background against the curated session stream."""

    def __init__(
        self,
        manager: SessionManager,
        compactor: TrajectoryCompactor,
        token_counter: TokenCounter | None = None,
        telemetry_store: TelemetryStore | None = None,
        repo_root: Path | None = None,
    ) -> None:
        self.manager = manager
        self.compactor = compactor
        self.token_counter = token_counter or TokenCounter()
        resolved_root = (repo_root or _default_repo_root()).expanduser().resolve()
        self.telemetry = CompactionTelemetry(
            kind="trajectory",
            store=telemetry_store or TelemetryStore(telemetry_session_path(manager.session_id)),
            logger=compaction_logger(resolved_root, name="quasipilot.trajectory", log_dir="trajectory"),
        )
        self._mutex = Lock()
        self._worker: Thread | None = None
        self._pending: PendingTrajectoryCompaction | None = None
        self._pending_path = _pending_trajectory_path(manager)
        self._pending = self._load_pending()

    def is_running(self) -> bool:
        with self._mutex:
            worker = self._worker
        return worker is not None and worker.is_alive()

    def prepare_for_agent(self) -> str:
        while True:
            with self._mutex:
                worker = self._worker
                pending = self._pending or self._load_pending()
                self._pending = pending
            if worker is not None and worker.is_alive():
                worker.join()
                continue
            if pending is not None:
                return self._apply_pending(pending)
            return "not_ready"

    def request_compaction(self) -> str:
        with self._mutex:
            worker = self._worker
            if worker is not None and worker.is_alive():
                return "running"
            if self._pending is not None:
                return "ready"
            snapshot = self.manager.read_curated()
            payload = self._payload(snapshot, trigger="trajectory", reason="turn_interval")
            if not self.compactor.should_compact(snapshot):
                self.telemetry.skip(payload)
                return "not_needed"
            latest_turn = max((event.turn for event in snapshot), default=0)
            self._worker = Thread(
                target=self._run_compaction,
                args=(snapshot, latest_turn),
                name=f"trajectory-compaction-{self.manager.session_id}",
                daemon=True,
            )
            self._worker.start()
            return "started"

    def _run_compaction(self, snapshot_events: list[SessionEvent], snapshot_latest_turn: int) -> None:
        payload = self._payload(snapshot_events, trigger="trajectory", reason="turn_interval")
        started = perf_counter()
        self.telemetry.start(payload)
        try:
            result = self.compactor.compact(snapshot_events)
            pending = PendingTrajectoryCompaction(
                result_events=result.events,
                snapshot_latest_turn=snapshot_latest_turn,
                payload=payload,
                compacted_event_count=result.compacted_event_count,
                compacted_turn_count=len(result.compacted_turns),
                result_event_count=len(result.events),
                model_names=result.model_names,
                token_usage=result.token_usage,
                duration_ms=int((perf_counter() - started) * 1000),
            )
            with self._mutex:
                self._pending = pending
                self._store_pending(pending)
        except Exception as exc:
            self.telemetry.error(
                {
                    **payload,
                    "error": str(exc),
                    "exception_type": type(exc).__name__,
                }
            )
        finally:
            with self._mutex:
                self._worker = None

    def _payload(self, events: list[SessionEvent], *, trigger: str, reason: str) -> dict[str, object]:
        session_events = list(events)
        turns = sorted({event.turn for event in session_events})
        return {
            "trigger": trigger,
            "reason": reason,
            "model_names": {
                "compactor": get_model_name(self.compactor.policy.compactor_model),
            },
            "counts": {
                "source_event_count": len(session_events),
                "turn_count": len(turns),
                "latest_turn": turns[-1] if turns else 0,
            },
            "token_usage": {
                "source_tokens": self.token_counter.count_events(session_events) if session_events else 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "call_count": 0,
            },
            **compaction_time_span(session_events),
        }

    def _apply_pending(self, pending: PendingTrajectoryCompaction) -> str:
        merged_events = self.manager.apply_compaction_result(
            pending.result_events,
            snapshot_latest_turn=pending.snapshot_latest_turn,
        )
        self.telemetry.end(
            {
                **pending.payload,
                "counts": {
                    **dict(pending.payload["counts"]),
                    "compacted_event_count": pending.compacted_event_count,
                    "compacted_turn_count": pending.compacted_turn_count,
                    "result_event_count": pending.result_event_count,
                    "curated_event_count": len(merged_events),
                },
                "model_names": pending.model_names,
                "token_usage": {
                    **dict(pending.payload["token_usage"]),
                    **pending.token_usage,
                },
                "duration_ms": pending.duration_ms,
            }
        )
        with self._mutex:
            if self._pending == pending:
                self._pending = None
            self._clear_pending()
        return "applied"

    def _store_pending(self, pending: PendingTrajectoryCompaction) -> None:
        self._pending_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._pending_path.with_suffix(self._pending_path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "result_events": [event.to_json_dict() for event in pending.result_events],
                    "snapshot_latest_turn": pending.snapshot_latest_turn,
                    "payload": pending.payload,
                    "compacted_event_count": pending.compacted_event_count,
                    "compacted_turn_count": pending.compacted_turn_count,
                    "result_event_count": pending.result_event_count,
                    "model_names": pending.model_names,
                    "token_usage": pending.token_usage,
                    "duration_ms": pending.duration_ms,
                },
                handle,
                ensure_ascii=False,
            )
        temp_path.replace(self._pending_path)

    def _load_pending(self) -> PendingTrajectoryCompaction | None:
        if not self._pending_path.exists():
            return None
        with self._pending_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return PendingTrajectoryCompaction(
            result_events=[SessionEvent.from_json_dict(item) for item in data["result_events"]],
            snapshot_latest_turn=int(data["snapshot_latest_turn"]),
            payload=dict(data["payload"]),
            compacted_event_count=int(data["compacted_event_count"]),
            compacted_turn_count=int(data["compacted_turn_count"]),
            result_event_count=int(data["result_event_count"]),
            model_names={str(key): str(value) for key, value in dict(data["model_names"]).items()},
            token_usage={str(key): int(value) for key, value in dict(data["token_usage"]).items()},
            duration_ms=int(data["duration_ms"]),
        )

    def _clear_pending(self) -> None:
        if self._pending_path.exists():
            self._pending_path.unlink()


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _pending_trajectory_path(manager: SessionManager) -> Path:
    base = manager.root or manager.curated_path.parent.parent
    return Path(base) / "pending_trajectories" / f"{manager.session_id}.json"
