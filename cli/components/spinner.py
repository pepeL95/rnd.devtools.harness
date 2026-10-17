from __future__ import annotations

import time

from rich.text import Text
from textual.widgets import Static

# Each sequence traces the letter's lower structure upward in one braille cell.
_DRAW_SEQUENCES = (
    ("p", ("⡀", "⡄", "⡆", "p")),
    ("p", ("⡀", "⡄", "⡆", "p")),
    ("a", ("⣀", "⣠", "⣶", "a")),
)
_HOLD_FRAMES = 4


def _build_wordmark_frames() -> tuple[str, ...]:
    completed = ""
    frames: list[str] = []
    for letter, strokes in _DRAW_SEQUENCES:
        frames.extend((completed + stroke).ljust(3) for stroke in strokes)
        completed += letter
    return tuple(frames)


_WORDMARK_FRAMES = _build_wordmark_frames()


class WorkingSpinner(Static):
    """Draw a muted PPA wordmark from bottom to top, then restart."""

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
        self._frame = (self._frame + 1) % (len(_WORDMARK_FRAMES) + _HOLD_FRAMES)
        self._refresh_label()

    def _refresh_label(self) -> None:
        elapsed = time.monotonic() - self._started
        self.update(self._build_label(elapsed))

    def _build_label(self, elapsed: float) -> Text:
        label = Text()
        label.append(self._wordmark(), style="dim")
        label.append(f"  {self._status} · {elapsed:0.1f}s", style="dim")
        return label

    def _wordmark(self) -> str:
        if self._frame >= len(_WORDMARK_FRAMES):
            return "ppa"
        return _WORDMARK_FRAMES[self._frame]

    def set_status(self, status: str) -> None:
        self._status = status
        self._refresh_label()
