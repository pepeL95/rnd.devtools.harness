from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkspaceState:
    session_id: str | None = None


class WorkspaceStore:
    """Persist the minimal workspace reference for the active session."""

    def __init__(self, cwd: str | Path | None = None) -> None:
        self._cwd = Path(cwd).expanduser().resolve() if cwd else Path.cwd().expanduser().resolve()

    @property
    def path(self) -> Path:
        return self._cwd / ".quasipilot" / "workspace.json"

    def ensure(self) -> Path:
        if not self.path.exists():
            self.write(WorkspaceState())
        return self.path

    def load(self) -> WorkspaceState:
        config = _read_workspace(self.path)
        raw = config.get("session_id")
        return WorkspaceState(session_id=str(raw) if raw else None)

    def write(self, state: WorkspaceState) -> None:
        _write_workspace(self.path, {"session_id": state.session_id})


def workspace_path(cwd: str | Path | None = None) -> Path:
    return WorkspaceStore(cwd).path


def ensure_workspace(cwd: str | Path | None = None) -> Path:
    return WorkspaceStore(cwd).ensure()


def load_session_id(cwd: str | Path | None = None) -> str | None:
    return WorkspaceStore(cwd).load().session_id


def save_session_id(session_id: str | None, cwd: str | Path | None = None) -> None:
    WorkspaceStore(cwd).write(WorkspaceState(session_id=session_id))


def _read_workspace(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        return {}
    return dict(data)


def _write_workspace(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temp_path.replace(path)
