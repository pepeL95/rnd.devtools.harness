from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from core.locks.base import FileLock, LockLease
from core.session.events import SessionEvent
from core.session.io import append_events, read_events, replace_events


@dataclass(frozen=True)
class SessionLease:
    token: str
    name: str
    snapshot_events: list[SessionEvent]


class SessionLock:
    """Session-specific lock around curated history and its pending overlay."""

    def __init__(self, curated_path: Path, lock_path: Path, pending_path: Path) -> None:
        self.curated_path = curated_path
        self.pending_path = pending_path
        self.file_lock = FileLock(lock_path)

    def read_curated(self, include_pending: bool = True) -> list[SessionEvent]:
        curated = read_events(self.curated_path)
        if not include_pending:
            return curated
        return [*curated, *self.read_pending()]

    def read_pending(self) -> list[SessionEvent]:
        return read_events(self.pending_path)

    def append(self, events: Iterable[SessionEvent]) -> None:
        target = self.pending_path if self.is_locked() else self.curated_path
        append_events(target, events)

    def replace(self, events: Iterable[SessionEvent]) -> None:
        replace_events(self.curated_path, events)

    def is_locked(self) -> bool:
        return self.file_lock.is_locked()

    def current(self) -> dict | None:
        return self.file_lock.current()

    def begin(self, name: str, metadata: dict | None = None) -> SessionLease | None:
        lease = self.file_lock.acquire(name=name, metadata=metadata)
        if lease is None:
            return None
        return SessionLease(
            token=lease.token,
            name=lease.name,
            snapshot_events=self.read_curated(include_pending=False),
        )

    def finalize(self, lease: SessionLease, compacted_events: Iterable[SessionEvent]) -> list[SessionEvent]:
        self._assert_owned(lease)
        merged = [*list(compacted_events), *self.read_pending()]
        replace_events(self.curated_path, merged)
        self._clear(lease)
        return merged

    def abort(self, lease: SessionLease) -> list[SessionEvent]:
        self._assert_owned(lease)
        merged = self.read_curated(include_pending=True)
        replace_events(self.curated_path, merged)
        self._clear(lease)
        return merged

    def _assert_owned(self, lease: SessionLease) -> None:
        self.file_lock.assert_owned(LockLease(token=lease.token, name=lease.name))

    def _clear(self, lease: SessionLease) -> None:
        if self.pending_path.exists():
            self.pending_path.unlink()
        self.file_lock.release(LockLease(token=lease.token, name=lease.name))
