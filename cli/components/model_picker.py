from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, ListItem, ListView

from cli.utilities.models import ModelSummary

_MODEL_ITEM_PREFIX = "model-"


def _item_dom_id(model_name: str) -> str:
    return f"{_MODEL_ITEM_PREFIX}{model_name.replace('.', '-').replace('/', '-')}"


def _model_name_from_dom_id(dom_id: str | None, models: list[ModelSummary]) -> str | None:
    if dom_id is None:
        return None
    for model in models:
        if _item_dom_id(model.name) == dom_id:
            return model.name
    return None


class ModelPickerScreen(ModalScreen[str | None]):
    """Modal picker for available driver models."""

    DEFAULT_CSS = """
    ModelPickerScreen {

    }

    #picker-panel {
        width: 90%;
        height: 70%;
        border: none;
        padding: 1 2;
    }

    #picker-list {
        height: 1fr;
        margin-top: 1;
        background: transparent;
    }

    .model-preview {
        color: $text-muted;
    }
    """

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, models: list[ModelSummary]) -> None:
        super().__init__()
        self._models = models

    def compose(self) -> ComposeResult:
        with Vertical(id="picker-panel"):
            yield Label("/models", markup=False)
            items = self._items()
            if items:
                yield ListView(*items, id="picker-list")
            else:
                yield Label("no models", classes="model-preview", markup=False)

    def _items(self) -> list[ListItem]:
        return [self._item(summary) for summary in self._models]

    def _item(self, summary: ModelSummary) -> ListItem:
        current = "* " if summary.is_current else "  "
        description = f"  {summary.description}" if summary.description else ""
        label = f"{current}{summary.name}{description}"
        return ListItem(Label(label, markup=False), id=_item_dom_id(summary.name))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(_model_name_from_dom_id(event.item.id, self._models))

    def action_cancel(self) -> None:
        self.dismiss(None)
