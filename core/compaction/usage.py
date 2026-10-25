from __future__ import annotations

from typing import Any

from core.compaction.token_counter import TokenCounter


class TokenUsage:
    """Aggregate provider usage, explicitly marking calls that require estimates."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.call_count = 0
        self.reported_calls = 0
        self.estimated_calls = 0

    def record(self, response: Any, prompt: str, text: str, counter: TokenCounter) -> None:
        metadata = getattr(response, "usage_metadata", None)
        keys = ("input_tokens", "output_tokens", "total_tokens")
        if isinstance(metadata, dict) and all(
            type(metadata.get(key)) is int and metadata[key] >= 0 for key in keys
        ):
            self.input_tokens += metadata["input_tokens"]
            self.output_tokens += metadata["output_tokens"]
            self.total_tokens += metadata["total_tokens"]
            self.reported_calls += 1
        else:
            input_tokens = counter.count_text(prompt)
            output_tokens = counter.count_text(text)
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.total_tokens += input_tokens + output_tokens
            self.estimated_calls += 1

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
            "reported_calls": self.reported_calls,
            "estimated_calls": self.estimated_calls,
        }
