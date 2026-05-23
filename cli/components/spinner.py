from __future__ import annotations

import time

from textual.widgets import Static

_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class WorkingSpinner(Static):
    """Codex-style working indicator with elapsed time."""

    DEFAULT_CSS = """
    WorkingSpinner {
        width: 100%;
        margin: 0 0 1 2;
        color: $text-muted;
    }
    """

    def __init__(self) -> None:
        super().__init__("", markup=False)
        self._started = time.monotonic()
        self._frame = 0
        self._timer = None

    def on_mount(self) -> None:
        self._refresh_label()
        self._timer = self.set_interval(0.08, self._tick)

    def on_unmount(self) -> None:
        if self._timer is not None:
            self._timer.stop()

    def _tick(self) -> None:
        self._frame = (self._frame + 1) % len(_FRAMES)
        self._refresh_label()

    def _refresh_label(self) -> None:
        elapsed = time.monotonic() - self._started
        frame = _FRAMES[self._frame]
        self.update(f"{frame} working · {elapsed:0.1f}s")
