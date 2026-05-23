from __future__ import annotations

from textual.widgets import Static


class Divider(Static):
    """Minimal section divider."""

    DEFAULT_CSS = """
    Divider {
        width: 100%;
        height: 1;
        margin: 1 0;
        color: $surface-lighten-1;
        content-align: center middle;
    }
    """

    def __init__(self) -> None:
        super().__init__("─" * 40)
