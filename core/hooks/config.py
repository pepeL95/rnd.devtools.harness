from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import tomllib

from pydantic import BaseModel, ConfigDict, Field, model_validator

Trigger = Literal[
    "before.turn", "after.turn", "before.model", "after.model",
    "before.tool", "after.tool", "before.command", "after.command",
]


class Match(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    tool: list[str] | None = None
    cmd: list[str] | None = None
    args: list[str] | None = None


class HookConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    id: str = Field(min_length=1)
    enabled: bool = True
    type: Literal["steering", "code"] | None = None
    trigger: Trigger | None = None
    file: str | None = None
    match: Match = Field(default_factory=Match)
    recursion: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_hook(self) -> HookConfig:
        if not self.enabled:
            return self
        if not self.type or not self.trigger or not self.file:
            raise ValueError("Enabled hooks require type, trigger, and file")
        if "recursion" in self.model_fields_set and self.trigger != "after.turn":
            raise ValueError("recursion is only valid for after.turn")
        fields = self.match.model_fields_set
        allowed = {"tool"} if self.trigger.endswith(".tool") else (
            {"cmd", "args"} if self.trigger.endswith(".command") else set()
        )
        if fields - allowed:
            raise ValueError(f"Invalid match fields for {self.trigger}: {fields - allowed}")
        return self


@dataclass(frozen=True)
class Hook:
    config: HookConfig
    path: Path


def load_hooks(cwd: Path, *, global_root: Path | None = None) -> tuple[Hook, ...]:
    roots = [global_root or Path.home() / ".quasipilot/.hooks", cwd / ".quasipilot/.hooks"]
    merged: dict[str, Hook] = {}
    for root in dict.fromkeys(path.expanduser().absolute() for path in roots):
        manifest = root / "hooks.toml"
        if not manifest.exists():
            continue
        try:
            document = tomllib.loads(manifest.read_text(encoding="utf-8"))
            if set(document) - {"hooks"} or not isinstance(document.get("hooks", []), list):
                raise ValueError("Expected only [[hooks]] entries")
            seen: set[str] = set()
            for entry in document.get("hooks", []):
                config = HookConfig.model_validate(entry)
                if config.id in seen:
                    raise ValueError(f"Duplicate hook ID: {config.id}")
                seen.add(config.id)
                merged.pop(config.id, None)
                if not config.enabled:
                    continue
                base = (root / str(config.type)).resolve()
                relative = Path(str(config.file))
                path = (base / relative).resolve()
                if relative.is_absolute() or not path.is_relative_to(base):
                    raise ValueError(f"Hook file must remain inside {base}")
                if not path.is_file():
                    raise ValueError(f"Hook file does not exist: {path}")
                merged[config.id] = Hook(config, path)
        except (ValueError, OSError) as exc:
            raise ValueError(f"Invalid hooks manifest {manifest}: {exc}") from exc
    return tuple(merged.values())
