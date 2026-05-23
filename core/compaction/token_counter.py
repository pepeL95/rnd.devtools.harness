from __future__ import annotations

from collections.abc import Iterable

from core.compaction.serialization import events_to_trajectory
from core.session.events import SessionEvent


class TokenCounter:
    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self.encoding_name = encoding_name

    def count_text(self, text: str) -> int:
        try:
            import tiktoken

            encoding = tiktoken.get_encoding(self.encoding_name)
            return len(encoding.encode(text))
        except Exception:
            return max(1, len(text) // 4)

    def count_events(self, events: Iterable[SessionEvent]) -> int:
        return self.count_text(events_to_trajectory(events))

