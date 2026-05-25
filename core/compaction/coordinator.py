from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any
from uuid import uuid4

from core.compaction.compactor import Compactor
from core.compaction.logging import CompactionLogger
from core.compaction.token_counter import TokenCounter
from core.session.events import SessionEvent
from core.session.manager import SessionManager

CompactionEventCallback = Callable[[str, dict[str, Any]], None]


class CompactionCoordinator:
    """Run full session compaction synchronously against the curated session."""

    def __init__(
        self,
        manager: SessionManager,
        compactor: Compactor,
        token_counter: TokenCounter | None = None,
        on_compaction_event: CompactionEventCallback | None = None,
        log_root: Path | None = None,
    ) -> None:
        self.manager = manager
        self.compactor = compactor
        self.token_counter = token_counter or TokenCounter()
        self.on_compaction_event = on_compaction_event
        self.logger = CompactionLogger(log_root or Path.cwd() / ".logs" / "compaction")
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
        run_id = uuid4().hex
        try:
            source_events = list(self.manager.read_curated(include_pending=True))
            if not source_events:
                self.logger.log(
                    self.manager.session_id,
                    run_id,
                    "skip",
                    "not_needed",
                    {
                        "trigger": trigger,
                        "reason": "empty_session",
                        "event_count": 0,
                        "content": "session compaction skipped: no events available",
                    },
                )
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
                    self.logger.log(
                        self.manager.session_id,
                        run_id,
                        "skip",
                        "not_needed",
                        {
                            "trigger": trigger,
                            "reason": decision.reason,
                            "estimated_tokens": estimated,
                            "trigger_tokens": self.compactor.policy.trigger_tokens,
                            "keep_last_turns": self.compactor.policy.keep_last_turns,
                            "event_count": len(source_events),
                            "latest_turn": latest_turn,
                            "content": f"session compaction skipped: {reason}",
                        },
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
            return "completed" if self._run_compaction(run_id, source_events, estimated, payload) else "failed"
        finally:
            self._run_lock.release()

    def _run_compaction(
        self,
        run_id: str,
        source_events: list[SessionEvent],
        estimated: int,
        payload: dict[str, Any],
    ) -> bool:
        start_payload = {**payload, "source_event_count": len(source_events)}
        self.logger.log(self.manager.session_id, run_id, "start", "running", start_payload)
        self._emit_compaction_event(run_id, "start", start_payload)
        previous_stage_handler = self.compactor.on_stage
        try:
            self.compactor.on_stage = lambda stage, stage_payload: self.logger.log(
                self.manager.session_id,
                run_id,
                "stage",
                "running",
                {"stage": stage, **stage_payload},
            )
            result = self.compactor.compact(source_events, token_estimate=estimated)
            rewrite_started = perf_counter()
            self.manager.replace_curated(result.events)
            rewrite_elapsed_ms = int((perf_counter() - rewrite_started) * 1000)
            self.logger.log(
                self.manager.session_id,
                run_id,
                "curated_rewrite",
                "completed",
                {
                    "curated_event_count": len(result.events),
                    "rewrite_elapsed_ms": rewrite_elapsed_ms,
                },
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
            self.logger.log(self.manager.session_id, run_id, "end", "completed", end_payload)
            self._emit_compaction_event(run_id, "end", end_payload)
            return True
        except Exception as exc:
            error_payload = {
                **payload,
                "error": str(exc),
                "exception_type": type(exc).__name__,
                "content": self._format_compaction_message("error", payload["trigger"], error=str(exc)),
            }
            self.logger.log(self.manager.session_id, run_id, "error", "failed", error_payload)
            self._emit_compaction_event(run_id, "error", error_payload)
            return False
        finally:
            self.compactor.on_stage = previous_stage_handler

    def _emit_compaction_event(self, run_id: str, phase: str, payload: dict[str, Any]) -> None:
        if self.on_compaction_event is None:
            return
        try:
            self.on_compaction_event(phase, payload)
        except Exception as exc:
            self.logger.log(
                self.manager.session_id,
                run_id,
                "ui_callback_error",
                "ignored",
                {
                    "phase": phase,
                    "error": str(exc),
                    "exception_type": type(exc).__name__,
                },
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
