from __future__ import annotations

from threading import Lock


STEERING_PREFIX = "The user has interrupted your session by providing the following instructions:"


class LiveSteeringInterrupt(Exception):
    """Raised to stop the active turn at the next tool boundary."""

    def __init__(self, steering: str) -> None:
        self.steering = steering
        super().__init__("live steering requested")


class LiveSteeringController:
    """Thread-safe queue for steering requests submitted during agent work."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._pending: list[str] = []

    def submit(self, text: str) -> None:
        note = text.strip()
        if not note:
            return
        with self._lock:
            self._pending.append(note)

    def drain(self) -> str | None:
        with self._lock:
            if not self._pending:
                return None
            pending = self._pending
            self._pending = []
        return "\n\n".join(pending)


def format_live_steering_message(steering: str) -> str:
    note = steering.strip()
    if not note:
        return STEERING_PREFIX
    return f"{STEERING_PREFIX} {note}"
