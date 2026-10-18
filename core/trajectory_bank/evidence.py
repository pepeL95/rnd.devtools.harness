from __future__ import annotations

import hashlib
import json
import re
from collections import Counter

from core.session.events import EventType, SessionEvent, is_pinned_instruction
from core.session.turns import display_history_events


def fingerprint(events: list[SessionEvent]) -> str:
    data = [event.to_json_dict() for event in events if not is_pinned_instruction(event)]
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def reasoning_chain(events: list[SessionEvent]) -> str:
    """Extract recorded reasoning, never tool results or synthetic harness hints."""
    parts: list[str] = []
    for event in events:
        payload = event.payload
        if event.type == EventType.REASONING and not payload.get("kind"):
            if payload.get("reasoning_format") not in {None, "reasoning", "thinking", "additional_kwargs"}:
                continue
            text = payload.get("content")
        elif event.type == EventType.TOOL and payload.get("name") == "reasoning":
            args = payload.get("args")
            text = args.get("reasoning") if isinstance(args, dict) else None
        else:
            continue
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n\n".join(parts)


def trajectory_document(events: list[SessionEvent], trajectory: str) -> str:
    """Ground a trajectory in real user messages and the last assistant response."""
    history = display_history_events(events)
    users = [_event_text(event) for event in history if event.type == EventType.USER]
    user = "\n\n".join(text for text in users if text) or "(No user text recorded.)"
    assistant = next((_event_text(event) for event in reversed(history)
                      if event.type == EventType.ASSISTANT), "") or "(No assistant response recorded.)"
    return f"User:\n{user}\n\nTrajectory:\n{trajectory}\n\nAssistant:\n{assistant}"


def _event_text(event: SessionEvent) -> str:
    content = event.payload.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(block["text"] for block in content
                         if isinstance(block, dict) and block.get("type") == "text"
                         and isinstance(block.get("text"), str)).strip()
    return ""


def keywords(content: str) -> list[str]:
    """Stable source-grounded terms; code identifiers precede ordinary vocabulary."""
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_./-]{2,}", content)
    counts = Counter(tokens)
    stop = {"the", "and", "that", "this", "with", "for", "from", "was", "were", "not", "content", "type", "role", "payload"}
    return sorted((word for word in counts if word.lower() not in stop),
                  key=lambda word: (not any(char in word for char in "_/ ."), -counts[word], word))[:12]
