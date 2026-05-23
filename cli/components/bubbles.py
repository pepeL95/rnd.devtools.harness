from __future__ import annotations

from textual.widgets import Static


class UserBubble(Static):
    """User message bubble."""

    DEFAULT_CSS = """
    UserBubble {
        width: 100%;
        margin: 1;
        padding: 0 1;
        background: $surface;
        color: $text;
        padding: 1;
    }
    """

    def __init__(self, text: str) -> None:
        super().__init__(text, markup=False)
        self.add_class("user-bubble")


class AIBubble(Static):
    """Assistant message bubble."""

    DEFAULT_CSS = """
    AIBubble {
        width: 100%;
        margin: 1;
        padding: 0 1;
        color: $text-muted;
    }
    """

    def __init__(self, text: str) -> None:
        super().__init__(text, markup=False)
        self.add_class("ai-bubble")
