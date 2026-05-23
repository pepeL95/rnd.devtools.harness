from __future__ import annotations

from textual.widgets import Static


class RuntimeBar(Static):
    """Footer showing model and cwd."""

    DEFAULT_CSS = """
    RuntimeBar {
        height: 1;
        padding: 0 1;
        color: $text-muted;
        background: $panel;
    }
    """

    def __init__(self, model: str, cwd: str) -> None:
        super().__init__(self._label(model, cwd))
        self._model = model
        self._cwd = cwd

    @staticmethod
    def _label(model: str, cwd: str) -> str:
        return f"{model}  ·  {cwd}"

    def update_runtime(self, model: str | None = None, cwd: str | None = None) -> None:
        if model is not None:
            self._model = model
        if cwd is not None:
            self._cwd = cwd
        self.update(self._label(self._model, self._cwd))
