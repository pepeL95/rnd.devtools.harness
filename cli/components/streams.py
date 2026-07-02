from __future__ import annotations

import json
from typing import Any

from rich.text import Text
from textual.widgets import Static


class ToolStream(Static):
    """Live tool invocation line."""

    DEFAULT_CSS = """
    ToolStream {
        width: 100%;
        height: auto;
        margin: 1 2;
    }
    """

    def __init__(
        self,
        name: str,
        input_text: str = "",
        output: Any = None,
        *,
        continuation: bool = False,
    ) -> None:
        super().__init__(self._build_content(name, input_text, output, continuation=continuation), markup=False)

    def _build_content(self, name: str, input_text: str, output: Any, *, continuation: bool = False) -> Text:
        text = Text()
        if not continuation:
            text.append("\u2022 ", style="#8BC4A3")
            text.append("[tool] ", style="bold #8BC4A3")
            text.append(name, style="bold")
            if input_text:
                text.append(" ")
                text.append(input_text)

        output_block = _truncate_output(_pretty_output(output))
        if output_block:
            if not continuation:
                text.append("\n")
            text.append(_format_output_block(output_block), style="dim")

        return text


class ReasonStream(Static):
    """Live reasoning snippet."""

    DEFAULT_CSS = """
    ReasonStream {
        width: 100%;
        padding: 0 2 1 2;
        color: $text-muted;
        text-style: italic;
    }
    """

    def __init__(self, text: str) -> None:
        super().__init__(text, markup=False)


def _pretty_json(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        try:
            parsed = json.loads(stripped)
        except Exception:
            return stripped
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    try:
        return json.dumps(value, indent=2, ensure_ascii=False)
    except Exception:
        return str(value)


def _truncate_output(text: str, limit: int = 600) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _indent_block(text: str, prefix: str = "  ") -> str:
    return "\n".join(f"{prefix}{line}" if line else prefix.rstrip() for line in text.splitlines())


def _pretty_output(value: Any) -> str:
    return _pretty_json(value)


def _format_output_block(text: str) -> str:
    lines = text.splitlines() or [text]
    shown = _summarize_output_lines(lines)
    formatted: list[str] = []
    for index, line in enumerate(shown):
        prefix = "  \u2514 " if index == 0 else "    "
        formatted.append(f"{prefix}{line}" if line else "")
    return "\n".join(formatted)


def _summarize_output_lines(lines: list[str], head: int = 3, tail: int = 1) -> list[str]:
    if len(lines) <= head + tail:
        return lines
    hidden = len(lines) - head - tail
    return [*lines[:head], f"\u2026 +{hidden} lines", "", *lines[-tail:]]
