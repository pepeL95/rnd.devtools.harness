from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from langchain_core.language_models.chat_models import BaseChatModel

from core.compaction.llms import (
    get_default_compactor_model,
    get_default_critic_model,
    get_default_task_extractor_model,
)
from core.session.events import SessionEvent


@dataclass(frozen=True)
class CompactionDecision:
    should_compact: bool
    reason: str | None = None


@dataclass(frozen=True)
class CompactionPolicy:
    trigger_tokens: int = 80_000
    keep_last_turns: int = 2
    max_critic_loops: int = 1
    task_extractor_model: BaseChatModel = field(default_factory=get_default_task_extractor_model)
    compactor_model: BaseChatModel = field(default_factory=get_default_compactor_model)
    critic_model: BaseChatModel = field(default_factory=get_default_critic_model)
    trigger_after: timedelta | None = None
    trigger_on_day_change: bool = False

    def should_compact(self, estimated_tokens: int) -> bool:
        return self.compaction_decision(estimated_tokens).should_compact

    def compaction_decision(
        self,
        estimated_tokens: int,
        events: list[SessionEvent] | None = None,
        now: datetime | None = None,
    ) -> CompactionDecision:
        if estimated_tokens >= self.trigger_tokens:
            return CompactionDecision(True, "tokens")
        if not events:
            return CompactionDecision(False)

        event_times = [_parse_timestamp(event.timestamp) for event in events]
        event_times = [item for item in event_times if item is not None]
        if not event_times:
            return CompactionDecision(False)

        current = _normalize_datetime(now or datetime.now(timezone.utc))
        oldest = min(event_times)
        latest = max(event_times)
        if self.trigger_after is not None and current - oldest >= self.trigger_after:
            return CompactionDecision(True, "age")
        if self.trigger_on_day_change and latest.date() < current.date():
            return CompactionDecision(True, "day_change")
        return CompactionDecision(False)

    def validate(self) -> None:
        if self.trigger_tokens <= 0:
            raise ValueError("trigger_tokens must be positive.")
        if self.keep_last_turns < 0:
            raise ValueError("keep_last_turns must be non-negative.")
        if self.max_critic_loops < 0:
            raise ValueError("max_critic_loops must be non-negative.")
        if self.trigger_after is not None and self.trigger_after <= timedelta(0):
            raise ValueError("trigger_after must be positive when set.")


def _parse_timestamp(value: str) -> datetime | None:
    try:
        return _normalize_datetime(datetime.fromisoformat(value))
    except ValueError:
        return None


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
