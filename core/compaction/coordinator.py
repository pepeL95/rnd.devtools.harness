from __future__ import annotations

from collections.abc import Callable
import logging
import datetime
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

from core.compaction.compactor import Compactor
from core.compaction.telemetry import CompactionTelemetry
from core.compaction.token_counter import TokenCounter
from core.session.events import SessionEvent
from core.session.manager import SessionManager
from core.telemetry.store import TelemetryStore, telemetry_session_path

CompactionEventCallback = Callable[[str, dict[str, Any]], None]


class CompactionCoordinator:
    """Run full session compaction synchronously against the curated session."""

    def __init__(
        self,
        manager: SessionManager,
        compactor: Compactor,
        token_counter: TokenCounter | None = None,
        on_compaction_event: CompactionEventCallback | None = None,
        telemetry_store: TelemetryStore | None = None,
        repo_root: Path | None = None,
    ) -> None:
        self.manager = manager
        self.compactor = compactor
        self.token_counter = token_counter or TokenCounter()
        self.on_compaction_event = on_compaction_event
        resolved_root = (repo_root or _default_repo_root()).expanduser().resolve()
        self.telemetry = CompactionTelemetry(telemetry_store or TelemetryStore(telemetry_session_path(manager.session_id)))
        self.logger = _compaction_logger(resolved_root)
        self._run_lock = Lock()

    def is_running(self) -> bool:
        return self._run_lock.locked()

    def request_manual_compaction(self, runtime: Any = None) -> str:
        return self._schedule(trigger="manual", runtime=runtime, force=True)

    def request_policy_compaction(self, runtime: Any = None) -> str:
        return self._schedule(trigger="policy", runtime=runtime, force=False)

    def _schedule(self, trigger: str, runtime: Any, force: bool) -> str:
        if not self._run_lock.acquire(blocking=False):
            return "running"
        try:
            source_events = list(self.manager.read_curated(include_pending=True))
            if not source_events:
                self.telemetry.skip(
                    trigger=trigger,
                    reason="empty_session",
                    event_count=0,
                    latest_turn=0,
                )
                self.logger.info("session compaction skipped: empty session trigger=%s", trigger)
                return "not_needed"
            latest_turn = max((event.turn for event in source_events), default=0)
            estimated = self.token_counter.count_events(source_events)
            reason = "manual"
            if not force:
                decision = self.compactor.policy.compaction_decision(
                    estimated,
                    events=source_events,
                    now=_runtime_now(runtime),
                )
                if not decision.should_compact:
                    reason = decision.reason or "not_needed"
                    self.telemetry.skip(
                        trigger=trigger,
                        reason=reason,
                        estimated_tokens=estimated,
                        event_count=len(source_events),
                        latest_turn=latest_turn,
                    )
                    self.logger.info(
                        "session compaction skipped: trigger=%s reason=%s estimated_tokens=%s event_count=%s",
                        trigger,
                        reason,
                        estimated,
                        len(source_events),
                    )
                    return "not_needed"
                reason = decision.reason or reason
            payload = {
                "trigger": trigger,
                "reason": reason,
                "estimated_tokens": estimated,
                "trigger_tokens": self.compactor.policy.trigger_tokens,
                "keep_last_turns": self.compactor.policy.keep_last_turns,
                "event_count": len(source_events),
                "latest_turn": latest_turn,
                "content": self._format_compaction_message("start", trigger, estimated=estimated),
            }
            return "completed" if self._run_compaction(source_events, estimated, payload) else "failed"
        finally:
            self._run_lock.release()

    def _run_compaction(
        self,
        source_events: list[SessionEvent],
        estimated: int,
        payload: dict[str, Any],
    ) -> bool:
        started = perf_counter()
        start_payload = {**payload, "source_event_count": len(source_events)}
        self.telemetry.start(
            trigger=str(payload["trigger"]),
            reason=str(payload["reason"]),
            estimated_tokens=estimated,
            event_count=len(source_events),
            latest_turn=int(payload["latest_turn"]),
        )
        self.logger.info(
            "session compaction started: trigger=%s reason=%s estimated_tokens=%s event_count=%s",
            payload["trigger"],
            payload["reason"],
            estimated,
            len(source_events),
        )
        self._emit_compaction_event("start", start_payload)
        previous_stage_handler = self.compactor.on_stage
        try:
            self.compactor.on_stage = lambda stage, stage_payload: self.logger.debug(
                "session compaction stage=%s details=%s",
                stage,
                stage_payload,
            )
            result = self.compactor.compact(source_events, token_estimate=estimated)
            rewrite_started = perf_counter()
            self.manager.replace_curated(result.events)
            rewrite_elapsed_ms = int((perf_counter() - rewrite_started) * 1000)
            duration_ms = int((perf_counter() - started) * 1000)
            self.logger.info(
                "session compaction rewrote curated: curated_event_count=%s rewrite_elapsed_ms=%s",
                len(result.events),
                rewrite_elapsed_ms,
            )
            end_payload = {
                **payload,
                "compacted_event_count": result.compacted_event_count,
                "retained_event_count": result.retained_event_count,
                "revisions": result.revisions,
                "memory_chars": len(result.memory_document),
                "result_event_count": len(result.events),
                "curated_event_count": len(result.events),
                "content": self._format_compaction_message(
                    "end",
                    payload["trigger"],
                    compacted_event_count=result.compacted_event_count,
                    retained_event_count=result.retained_event_count,
                ),
            }
            self.telemetry.end(
                trigger=str(payload["trigger"]),
                estimated_tokens=estimated,
                compacted_event_count=result.compacted_event_count,
                retained_event_count=result.retained_event_count,
                curated_event_count=len(result.events),
                revisions=result.revisions,
                duration_ms=duration_ms,
            )
            self.logger.info(
                "session compaction finished: trigger=%s compacted=%s retained=%s curated=%s duration_ms=%s",
                payload["trigger"],
                result.compacted_event_count,
                result.retained_event_count,
                len(result.events),
                duration_ms,
            )
            self._emit_compaction_event("end", end_payload)
            return True
        except Exception as exc:
            error_payload = {
                **payload,
                "error": str(exc),
                "exception_type": type(exc).__name__,
                "content": self._format_compaction_message("error", payload["trigger"], error=str(exc)),
            }
            self.telemetry.error(
                trigger=str(payload["trigger"]),
                estimated_tokens=estimated,
                error=str(exc),
                exception_type=type(exc).__name__,
            )
            self.logger.exception(
                "session compaction failed: trigger=%s estimated_tokens=%s",
                payload["trigger"],
                estimated,
            )
            self._emit_compaction_event("error", error_payload)
            return False
        finally:
            self.compactor.on_stage = previous_stage_handler

    def _emit_compaction_event(self, phase: str, payload: dict[str, Any]) -> None:
        if self.on_compaction_event is None:
            return
        try:
            self.on_compaction_event(phase, payload)
        except Exception as exc:
            self.logger.warning(
                "session compaction UI callback failed: phase=%s error_type=%s error=%s",
                phase,
                type(exc).__name__,
                exc,
            )

    def _format_compaction_message(
        self,
        phase: str,
        trigger: str,
        *,
        estimated: int | None = None,
        compacted_event_count: int | None = None,
        retained_event_count: int | None = None,
        error: str | None = None,
    ) -> str:
        prefix = "manual" if trigger == "manual" else "policy"
        if phase == "start":
            suffix = f" ({estimated} estimated tokens)" if isinstance(estimated, int) else ""
            return f"{prefix} compaction started{suffix}"
        if phase == "end":
            details: list[str] = []
            if isinstance(compacted_event_count, int):
                details.append(f"{compacted_event_count} compacted")
            if isinstance(retained_event_count, int):
                details.append(f"{retained_event_count} retained")
            suffix = f" ({', '.join(details)})" if details else ""
            return f"{prefix} compaction finished{suffix}"
        if error:
            return f"{prefix} compaction failed: {error}"
        return f"{prefix} compaction failed"


def _runtime_now(runtime: Any) -> Any:
    context = getattr(runtime, "context", None)
    if isinstance(context, dict):
        return context.get("now")
    return getattr(context, "now", None)


def _compaction_logger(repo_root: Path) -> logging.Logger:
    logger = logging.getLogger("quasipilot.compaction")
    logger.setLevel(logging.DEBUG)
    log_filename = datetime.datetime.now().strftime("%m-%d-%Y.log")
    log_path = (repo_root / ".logs" / "compaction" / log_filename).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve() == log_path:
            return logger

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
