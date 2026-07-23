from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from core.session.events import RuntimeSnapshot


def workspace_config_path(cwd: str | Path | None = None) -> Path:
    base = Path(cwd).expanduser().resolve() if cwd else Path.cwd().expanduser().resolve()
    return base / ".quasipilot" / "workspace.json"


def global_workspace_config_path() -> Path:
    return Path.home() / ".quasipilot" / "workspace.json"


def ensure_local_workspace(cwd: str | Path | None = None) -> Path:
    config_path = workspace_config_path(cwd)
    if not config_path.exists():
        _write_config(config_path, {})
    return config_path


def load_session_id(cwd: str | Path | None = None) -> str | None:
    config = _read_config(workspace_config_path(cwd))
    raw = config.get("session_id")
    return str(raw) if raw else None


def load_python_interpreter(cwd: str | Path | None = None) -> Path | None:
    local_value = _load_python_interpreter_from_path(workspace_config_path(cwd))
    if local_value is not None:
        return local_value
    return _load_python_interpreter_from_path(global_workspace_config_path())


def save_python_interpreter(path: str | Path, cwd: str | Path | None = None) -> Path:
    interpreter = Path(path).expanduser().resolve()
    config_path = workspace_config_path(cwd)
    config = _read_config(config_path)
    runtime = dict(config.get("runtime") or {})
    runtime["python_interpreter"] = str(interpreter)
    config["runtime"] = runtime
    config.pop("python_interpreter", None)
    _write_config(config_path, config)
    return interpreter


def save_runtime_context(
    *,
    session_id: str | None,
    session_title: str | None,
    session_date: str | None,
    model_name: str | None,
    runtime: RuntimeSnapshot,
    cwd: str | Path | None = None,
) -> None:
    config_path = workspace_config_path(cwd)
    config = _read_config(config_path)
    config["session_id"] = session_id
    config["session_title"] = session_title
    config["session_date"] = session_date
    config["model"] = model_name
    config["runtime"] = asdict(runtime)
    _write_config(config_path, config)


def _load_python_interpreter_from_path(path: Path) -> Path | None:
    config = _read_config(path)
    runtime = config.get("runtime")
    raw = runtime.get("python_interpreter") if isinstance(runtime, dict) else None
    if not raw:
        return None
    return Path(str(raw)).expanduser().resolve()


def _read_config(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        return {}
    return dict(data)


def _write_config(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temp_path.replace(path)
