from __future__ import annotations

from textual.widgets import Input


class ChatInput(Input):
    """Bottom chat input bar."""

    DEFAULT_CSS = """
    ChatInput:ansi {
        margin: 0 1;
        padding: 1;
        border: none;
        background: #272c34;
    }
    ChatInput:focus {
        border: none;
        background: #272c34;
    }
    
    """

    def __init__(self) -> None:
        super().__init__(placeholder="› Type a message or /command", id="chat-input")
        self.cursor_blink = False
