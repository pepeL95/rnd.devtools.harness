from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, ListItem, ListView

from cli.utilities.sessions import SessionSummary

_SESSION_ITEM_PREFIX = "session-"
_DIVIDER_DOM_ID = "session-divider"


def _item_dom_id(session_id: str) -> str:
    """Map session id to a valid Textual DOM identifier."""
    return f"{_SESSION_ITEM_PREFIX}{session_id}"


def _session_id_from_dom_id(dom_id: str | None) -> str | None:
    if dom_id and dom_id.startswith(_SESSION_ITEM_PREFIX):
        return dom_id[len(_SESSION_ITEM_PREFIX) :]
    return None


class SessionPickerScreen(ModalScreen[str | None]):
    """Multiselect-style session picker ordered by recent activity."""

    DEFAULT_CSS = """
    SessionPickerScreen {
        
    }

    #picker-panel {
        width: 90%;
        height: 70%;
        border: none;
        padding: 1 2;
    }

    #picker-panel:focus {
        border: none;
    }

    #picker-list {
        height: 1fr;
        margin-top: 1;
        background: transparent;
    }

    .session-preview {
        color: $text-muted;
    }

    .session-divider {
        color: $text-muted;
    }
    """

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, sessions: list[SessionSummary]) -> None:
        super().__init__()
        self._sessions = sessions

    def compose(self) -> ComposeResult:
        with Vertical(id="picker-panel"):
            yield Label("/sessions", markup=False)
            items = self._items()
            if items:
                yield ListView(*items, id="picker-list")
            else:
                yield Label("no sessions", classes="session-preview", markup=False)

    def _items(self) -> list[ListItem]:
        items = [self._item(summary) for summary in self._sessions]
        sponsored_count = 0
        for summary in self._sessions:
            if not summary.cwd_matches:
                break
            sponsored_count += 1
        if 0 < sponsored_count < len(items):
            items.insert(sponsored_count, self._divider_item())
        return items

    def _item(self, summary: SessionSummary) -> ListItem:
        stamp = summary.modified_at.astimezone().strftime("%Y-%m-%d %H:%M")
        preview = summary.preview or "—"
        label = f"{stamp}  {summary.session_id[:8]}  {preview}"
        return ListItem(Label(label, markup=False), id=_item_dom_id(summary.session_id))

    def _divider_item(self) -> ListItem:
        return ListItem(
            Label("──────────", classes="session-divider", markup=False),
            id=_DIVIDER_DOM_ID,
            disabled=True,
        )

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(_session_id_from_dom_id(event.item.id))

    def action_cancel(self) -> None:
        self.dismiss(None)
