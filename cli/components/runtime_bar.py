from __future__ import annotations

from textual.widgets import Static


class RuntimeBar(Static):
    """Footer showing model and cwd."""

    DEFAULT_CSS = """
    RuntimeBar {
        height: 1;
        padding: 0 0 0 2;
        color: $text-muted;
        background: transparent;
    }
    """

    def __init__(self, model: str, cwd: str, status: str | None = None) -> None:
        self._model = model
        self._cwd = cwd
        self._status = status
        super().__init__(self._label(model, cwd), markup=False)

    def _label(self, model: str, cwd: str) -> str:
        suffix = f"  ·  {self._status}" if self._status else ""
        return f"{model}  ·  {cwd}{suffix}"

    def update_runtime(
        self,
        model: str | None = None,
        cwd: str | None = None,
        status: str | None = None,
    ) -> None:
        if model is not None:
            self._model = model
        if cwd is not None:
            self._cwd = cwd
        self._status = status
        self.update(self._label(self._model, self._cwd))
