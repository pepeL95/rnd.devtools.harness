from __future__ import annotations

from textual import events
from textual.message import Message
from textual.widgets import TextArea


class ChatInput(TextArea):
    """Bottom chat input bar."""

    class Submitted(Message):
        """Posted when the user submits the current chat draft."""

        def __init__(self, input: ChatInput, value: str) -> None:
            self.input = input
            self.value = value
            super().__init__()

    DEFAULT_CSS = """
    ChatInput:ansi {
        margin: 0 1;
        padding: 1;
        border: none;
        background: #272c34;
        height: auto;
        min-height: 3;
        max-height: 10;
        scrollbar-background: transparent;
        scrollbar-color: $text-muted;
        scrollbar-size: 0 1;
    }
    ChatInput:focus {
        border: none;
        background: #272c34;
    }

    """

    def __init__(self) -> None:
        super().__init__(
            placeholder="› Type a message or /command",
            id="chat-input",
            compact=True,
            soft_wrap=True,
            show_line_numbers=False,
            highlight_cursor_line=False,
        )
        self.cursor_blink = False

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self, self.text))
            return

        if event.key == "ctrl+j":
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return

        await super()._on_key(event)
