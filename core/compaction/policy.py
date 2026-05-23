from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompactionPolicy:
    trigger_tokens: int = 8000
    keep_last_turns: int = 5
    max_critic_loops: int = 2
    model: str = "google_genai:gemini-3.5-flash"

    def should_compact(self, estimated_tokens: int) -> bool:
        return estimated_tokens >= self.trigger_tokens

    def validate(self) -> None:
        if self.trigger_tokens <= 0:
            raise ValueError("trigger_tokens must be positive.")
        if self.keep_last_turns < 0:
            raise ValueError("keep_last_turns must be non-negative.")
        if self.max_critic_loops < 0:
            raise ValueError("max_critic_loops must be non-negative.")
