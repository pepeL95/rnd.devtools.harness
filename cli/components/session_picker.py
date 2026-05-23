from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, ListItem, ListView

from cli.utilities.sessions import SessionSummary


class SessionPickerScreen(ModalScreen[str | None]):
    """Multiselect-style session picker ordered by recent activity."""

    DEFAULT_CSS = """
    SessionPickerScreen {
        align: center middle;
    }

    #picker-panel {
        width: 80%;
        max-width: 90;
        height: 70%;
        border: tall $primary;
        background: $panel;
        padding: 1 2;
    }

    #picker-list {
        height: 1fr;
        margin-top: 1;
    }

    .session-preview {
        color: $text-muted;
    }
    """

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, sessions: list[SessionSummary]) -> None:
        super().__init__()
        self._sessions = sessions

    def compose(self) -> ComposeResult:
        with Vertical(id="picker-panel"):
            yield Label("/sessions")
            items = [self._item(summary) for summary in self._sessions]
            if items:
                yield ListView(*items, id="picker-list")
            else:
                yield Label("no sessions", classes="session-preview")

    def _item(self, summary: SessionSummary) -> ListItem:
        stamp = summary.modified_at.astimezone().strftime("%Y-%m-%d %H:%M")
        preview = summary.preview or "—"
        label = f"{stamp}  {summary.session_id[:8]}  {preview}"
        return ListItem(Label(label), id=summary.session_id)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        session_id = event.item.id or None
        self.dismiss(session_id)

    def action_cancel(self) -> None:
        self.dismiss(None)
