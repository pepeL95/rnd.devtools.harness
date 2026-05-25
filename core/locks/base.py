from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from core.session.io import read_json, write_json_exclusive


@dataclass(frozen=True)
class LockLease:
    token: str
    name: str


class FileLock:
    """Small file-backed lease primitive for coordinated workflows."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def is_locked(self) -> bool:
        return self.path.exists()

    def current(self) -> dict[str, Any] | None:
        return read_json(self.path)

    def acquire(self, name: str, metadata: dict[str, Any] | None = None) -> LockLease | None:
        token = uuid4().hex
        payload = {"token": token, "name": name, **(metadata or {})}
        created = write_json_exclusive(self.path, payload)
        if not created:
            return None
        return LockLease(token=token, name=name)

    def assert_owned(self, lease: LockLease) -> None:
        current = self.current()
        if current is None or current.get("token") != lease.token:
            raise RuntimeError("Lock lease is no longer active.")

    def release(self, lease: LockLease) -> None:
        self.assert_owned(lease)
        if self.path.exists():
            self.path.unlink()
