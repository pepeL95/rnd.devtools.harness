from __future__ import annotations

from textual.widgets import Static


class Divider(Static):
    """Minimal section divider."""

    DEFAULT_CSS = """
    Divider {
        height: 1;
        margin: 1 2;
        color: $surface-lighten-1;
    }
    """

    def __init__(self) -> None:
        super().__init__("─" * 200)
