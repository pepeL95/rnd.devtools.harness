from __future__ import annotations

from typing import Any


def content_to_plaintext(content: Any) -> str:
    """Normalize session or message content for literal Textual display."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text") or block.get("reasoning")
                if text:
                    parts.append(str(text))
            elif block is not None:
                parts.append(str(block))
        if parts:
            return "\n".join(parts)
    if content is None:
        return ""
    return str(content)
