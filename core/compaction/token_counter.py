from __future__ import annotations

import json
from collections.abc import Iterable

from core.compaction.serialization import events_to_trajectory
from core.session.events import SessionEvent
from core.session.manager import reconstruct_agent_messages


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

    def count_context(self, events: Iterable[SessionEvent]) -> int:
        """Estimate restored message content, not storage metadata or provider framing."""
        total = 0
        for message in reconstruct_agent_messages(list(events)):
            content = message.content
            if content:
                total += self.count_text(content if isinstance(content, str)
                                         else json.dumps(content, ensure_ascii=False))
            calls = getattr(message, "tool_calls", None)
            if calls:
                total += self.count_text(json.dumps(calls, ensure_ascii=False))
        return total
