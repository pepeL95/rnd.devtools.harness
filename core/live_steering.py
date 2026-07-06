from __future__ import annotations

from threading import Lock



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




STEERING_INTROSPECTION = (
    "The user has interrupted me with new guidance. I should first decide whether they are "
    "replacing the task or steering the current one. Unless they clearly abandoned the "
    "original task, I should preserve it and treat the new message as refinement or "
    "constraint. I should keep both the original objective and the new guidance in mind, "
    "carry forward useful work, and continue without restarting unnecessarily."
)


def format_steering_introspection(steering: str) -> str:
    """Concise first-person pivot note injected after a live steering interrupt."""
    return STEERING_INTROSPECTION


class CancellationInterrupt(Exception):
    """Raised to abort the active turn at the next tool boundary."""


CANCELLATION_INTROSPECTION = (
    "The user cancelled this task mid-execution. I should reflect on why: perhaps "
    "I was diverging from their intent, they changed their mind, or the approach "
    "felt wrong. I will stop here — no further action — and await new direction."
)


def format_cancellation_introspection() -> str:
    """Compact first-person reflection injected when the user cancels a turn."""
    return CANCELLATION_INTROSPECTION
