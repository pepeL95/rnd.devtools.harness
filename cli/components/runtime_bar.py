from __future__ import annotations

from rich.text import Text
from textual.widgets import Static


class RuntimeBar(Static):
    """Footer showing model and cwd."""

    DEFAULT_CSS = """
    RuntimeBar {
        height: 1;
        padding: 0 0 0 2;
        color: $text-muted;
        background: transparent;
        text-style: none;
    }
    """

    def __init__(self, model: str, cwd: str, status: str | None = None) -> None:
        self._model = model
        self._cwd = cwd
        self._status = status
        self._curated_path: str | None = None
        super().__init__(self._label(model, cwd))

    def _label(self, model: str, cwd: str) -> Text:
        label = Text(f"{model}  ·  {cwd}", style="muted")
        if self._curated_path:
            label.append(f"  ·  {self._curated_path}")
        if self._status:
            label.append(f"  ·  {self._status}")
        return label

    def update_runtime(
        self,
        model: str | None = None,
        cwd: str | None = None,
        curated_path: str | None = None,
        status: str | None = None,
    ) -> None:
        if model is not None:
            self._model = model
        if cwd is not None:
            self._cwd = cwd
        self._curated_path = curated_path
        self._status = status
        self.update(self._label(self._model, self._cwd))
