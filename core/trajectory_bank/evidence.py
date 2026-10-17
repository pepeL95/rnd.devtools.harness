from __future__ import annotations

import hashlib
import json
import re
from collections import Counter

from core.session.events import SessionEvent, is_pinned_instruction


def fingerprint(events: list[SessionEvent]) -> str:
    data = [event.to_json_dict() for event in events if not is_pinned_instruction(event)]
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def keywords(content: str) -> list[str]:
    """Stable source-grounded terms; code identifiers precede ordinary vocabulary."""
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_./-]{2,}", content)
    counts = Counter(tokens)
    stop = {"the", "and", "that", "this", "with", "for", "from", "was", "were", "not", "content", "type", "role", "payload"}
    return sorted((word for word in counts if word.lower() not in stop),
                  key=lambda word: (not any(char in word for char in "_/ ."), -counts[word], word))[:12]
