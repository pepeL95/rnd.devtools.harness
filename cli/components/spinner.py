from __future__ import annotations

import time

from rich.text import Text
from textual.widgets import Static

_WORDMARK = "ppa"


class WorkingSpinner(Static):
    """Animated PPA wordmark with status and elapsed time."""

    DEFAULT_CSS = """
    WorkingSpinner {
        width: 100%;
        margin: 0 0 1 2;
        color: $text;
    }
    """

    def __init__(self, status: str = "working") -> None:
        super().__init__("", markup=False)
        self._started = time.monotonic()
        self._frame = 0
        self._timer = None
        self._status = status

    def on_mount(self) -> None:
        self._refresh_label()
        self._timer = self.set_interval(0.16, self._tick)

    def on_unmount(self) -> None:
        if self._timer is not None:
            self._timer.stop()

    def _tick(self) -> None:
        self._frame = (self._frame + 1) % len(_WORDMARK)
        self._refresh_label()

    def _refresh_label(self) -> None:
        elapsed = time.monotonic() - self._started
        self.update(self._build_label(elapsed))

    def _build_label(self, elapsed: float) -> Text:
        label = Text()
        for index, letter in enumerate(_WORDMARK):
            style = "bold" if index == self._frame else "dim"
            label.append(letter, style=style)
        label.append(f"  {self._status} · {elapsed:0.1f}s", style="dim")
        return label

    def set_status(self, status: str) -> None:
        self._status = status
        self._refresh_label()
