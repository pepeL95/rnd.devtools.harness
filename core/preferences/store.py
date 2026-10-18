from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import Lock
from uuid import uuid4


@dataclass(frozen=True)
class PreferencesSnapshot:
    path: Path
    exists: bool
    content: str
    revision: str | None


class PreferencesStore:
    """Read and atomically update workspace-local preferences."""

    def __init__(self, cwd: str | Path) -> None:
        self.path = Path(cwd).expanduser().resolve() / ".quasipilot" / "PREFERENCES.md"
        self._lock = Lock()

    def read(self) -> PreferencesSnapshot:
        with self._lock:
            return self._read_unlocked()

    def update(self, content: str, *, expected_revision: str | None) -> PreferencesSnapshot:
        with self._lock:
            current = self._read_unlocked()
            if current.revision != expected_revision:
                raise PreferencesConflictError(current.revision)

            self.path.parent.mkdir(parents=True, exist_ok=True)
            normalized = content.strip()
            temp_path = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
            try:
                temp_path.write_text(normalized + ("\n" if normalized else ""), encoding="utf-8")
                temp_path.replace(self.path)
            finally:
                temp_path.unlink(missing_ok=True)
            return self._read_unlocked()

    def _read_unlocked(self) -> PreferencesSnapshot:
        try:
            content = self.path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            legacy = self.path.with_name("DEVPROFILE.md")
            if legacy.exists():
                raise FileNotFoundError(
                    f"DEVPROFILE.md has been renamed to PREFERENCES.md. Move {legacy} to {self.path} "
                    "to preserve your existing preferences."
                ) from None
            return PreferencesSnapshot(path=self.path, exists=False, content="", revision=None)
        return PreferencesSnapshot(
            path=self.path,
            exists=True,
            content=content,
            revision=_revision(content),
        )


class PreferencesConflictError(RuntimeError):
    def __init__(self, current_revision: str | None) -> None:
        self.current_revision = current_revision
        super().__init__("PREFERENCES.md changed after it was read; read it again before retrying.")


def _revision(content: str) -> str:
    return f"sha256:{sha256(content.encode('utf-8')).hexdigest()}"
