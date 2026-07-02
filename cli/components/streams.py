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
        color: #C3E7CA;
    }
    """

    def __init__(self, name: str, args: Any = None, output: Any = None) -> None:
        super().__init__(self._build_content(name, args, output), markup=False)

    def _build_content(self, name: str, args: Any, output: Any) -> Text:
        text = Text()
        text.append("[tool] ", style="bold")
        text.append(name, style="bold")
        args_block = _pretty_json(args)
        if args_block:
            text.append("\n")
            text.append(args_block)
        output_block = _pretty_json(output)
        if output_block:
            text.append("\n")
            text.append(output_block, style="dim")
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
