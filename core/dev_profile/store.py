from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import Lock
from uuid import uuid4


@dataclass(frozen=True)
class DevProfileSnapshot:
    path: Path
    exists: bool
    content: str
    revision: str | None


class DevProfileStore:
    """Read and atomically update a workspace-local developer profile."""

    def __init__(self, cwd: str | Path) -> None:
        self.path = Path(cwd).expanduser().resolve() / ".quasipilot" / "DEVPROFILE.md"
        self._lock = Lock()

    def read(self) -> DevProfileSnapshot:
        with self._lock:
            return self._read_unlocked()

    def update(self, content: str, *, expected_revision: str | None) -> DevProfileSnapshot:
        with self._lock:
            current = self._read_unlocked()
            if current.revision != expected_revision:
                raise DevProfileConflictError(current.revision)

            self.path.parent.mkdir(parents=True, exist_ok=True)
            normalized = content.strip()
            temp_path = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
            try:
                temp_path.write_text(normalized + ("\n" if normalized else ""), encoding="utf-8")
                temp_path.replace(self.path)
            finally:
                temp_path.unlink(missing_ok=True)
            return self._read_unlocked()

    def _read_unlocked(self) -> DevProfileSnapshot:
        if not self.path.is_file():
            return DevProfileSnapshot(path=self.path, exists=False, content="", revision=None)
        content = self.path.read_text(encoding="utf-8").strip()
        return DevProfileSnapshot(
            path=self.path,
            exists=True,
            content=content,
            revision=_revision(content),
        )


class DevProfileConflictError(RuntimeError):
    def __init__(self, current_revision: str | None) -> None:
        self.current_revision = current_revision
        super().__init__("DEVPROFILE.md changed after it was read; read it again before retrying.")


def _revision(content: str) -> str:
    return f"sha256:{sha256(content.encode('utf-8')).hexdigest()}"
