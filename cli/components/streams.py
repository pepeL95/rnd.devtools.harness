from __future__ import annotations

import json
from typing import Any

from rich.console import Group
from rich.panel import Panel
from rich.syntax import Syntax
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

    def __init__(self, name: str, args: Any = None, output: Any = None) -> None:
        super().__init__(self._build_content(name, args, output), markup=False)

    def _build_content(self, name: str, args: Any, output: Any) -> Group:
        header = Text()
        header.append("[tool] ", style="bold #8BC4A3")
        header.append(name, style="bold")

        renderables: list[Any] = [header]

        args_block = _pretty_json(args)
        if args_block:
            renderables.append(
                Panel(
                    Syntax(args_block, "json", theme="monokai", word_wrap=True, line_numbers=False),
                    title="args",
                    title_align="left",
                    border_style="#3c5f50",
                    padding=(0, 1),
                )
            )

        output_block = _pretty_json(output)
        if output_block:
            renderables.append(
                Panel(
                    Syntax(output_block, "json", theme="monokai", word_wrap=True, line_numbers=False),
                    title="output",
                    title_align="left",
                    border_style="dim",
                    padding=(0, 1),
                )
            )

        return Group(*renderables)


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
