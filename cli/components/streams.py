from __future__ import annotations

import json
import random
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
    ) -> None:
        self._tool_name = name
        self._tool_input_text = input_text
        self._tool_output = output
        super().__init__(self._build_content(name, input_text, output), markup=False)

    def set_output(self, output: Any) -> None:
        self._tool_output = output
        self.update(self._build_content(self._tool_name,
                    self._tool_input_text, self._tool_output))

    def _build_content(self, name: str, input_text: str, output: Any) -> Text:
        text = Text()
        text.append(self._pretty_name(name), style="bold #8BC4A3")
        if input_text:
            text.append(" ")
            text.append(_format_input_block(name, input_text))

        output_block = _truncate_output(_pretty_output(output))
        if output_block:
            text.append("\n")
            text.append(_format_output_block(name, output_block), style="dim")

        return text

    def _pretty_name(self, name: str) -> str:
        if name == "execute":
            return random.choice(["Dispatched", "Yeeted", "Ran", "Slammed", "Ramrodded"])
        if name == "read_file":
            return "Read"
        if name == "write_file" or name == "make_file":
            return "Created"
        if name == "edit_file":
            return "Edited"
        if name == "glob":
            return "Searched"
        if name == "grep":
            return "Grepped"
        if name == "ls":
            return "Listed files"
        if name == "reasoning":
            return "Pondering"
        return name


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


def _pretty_output(value: Any) -> str:
    return _pretty_json(value)

def _format_input_block(name: str, text: str, limit: int = 400) -> str:
    if name == "reasoning":
        return text
    return _truncate_output(text, limit)

def _format_output_block(name: str, text: str) -> str:
    ignore_names = {"reasoning"}
    if name in ignore_names:
        return ""

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
