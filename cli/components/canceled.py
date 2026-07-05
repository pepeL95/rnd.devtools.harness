from __future__ import annotations

from textual.widgets import Static


class CanceledMessage(Static):
    """Muted message indicating a canceled request."""

    DEFAULT_CSS = """
    CanceledMessage {
        width: 100%;
        margin: 1 0;
        padding: 0 1;
        color: $text-muted;
        text-align: center;
    }
    """

    def __init__(self) -> None:
        super().__init__("[-------- canceled request --------]")
