from __future__ import annotations

from textual.widgets import Static


class ToolStream(Static):
    """Live tool invocation line."""

    DEFAULT_CSS = """
    ToolStream {
        width: 100%;
        margin: 1 2;
        color: #C3E7CA;
    }
    """

    def __init__(self, name: str, args: str = "") -> None:
        args = args.split("=") if args else []
        args = args[1].strip() if len(args) > 1 else ""
        if args.startswith("'") and args.endswith("'"):
            args = args[1:-1]
        label = f"[b]\[tool] {name}[/] [i]{args}[/]".strip()
        super().__init__(label, markup=True)


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
