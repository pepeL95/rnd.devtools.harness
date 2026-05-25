from __future__ import annotations

from textual.widgets import Static


class ToolStream(Static):
    """Live tool invocation line."""

    DEFAULT_CSS = """
    ToolStream {
        width: 100%;
        margin: 0 0 0 2;
        color: $warning;
    }
    """

    def __init__(self, name: str, args: str = "") -> None:
        label = f"{name} {args}".strip()
        super().__init__(label, markup=False)


class ReasonStream(Static):
    """Live reasoning snippet."""

    DEFAULT_CSS = """
    ReasonStream {
        width: 100%;
        padding: 1;
        color: $text-muted;
        text-style: italic;
    }
    """

    def __init__(self, text: str) -> None:
        super().__init__(text, markup=False)
