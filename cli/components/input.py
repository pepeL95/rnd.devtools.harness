from __future__ import annotations

from textual.widgets import Input


class ChatInput(Input):
    """Bottom chat input bar."""

    DEFAULT_CSS = """
    ChatInput {
        margin: 0 1 0 1;
        height: 3;
        border: tall $primary;
    }
    """

    def __init__(self) -> None:
        super().__init__(placeholder="message or /command", id="chat-input")
