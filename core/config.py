from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.utilities.workspace import workspace_directory


ReasoningEagerness = Literal["low", "medium", "high"]


class ReasoningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    eagerness: ReasoningEagerness = "high"


class SessionCompactionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trigger_tokens: int = Field(default=80_000, gt=0, strict=True)
    keep_last_turns: int = Field(default=2, ge=0, strict=True)
    max_critic_loops: int = Field(default=1, ge=0, strict=True)


class TrajectoryCompactionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trigger_every_turns: int = Field(default=1, gt=0, strict=True)


class CompactionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    offloading: Literal["lfs", "none"] | list[Literal["lfs"]] = Field(default_factory=lambda: ["lfs"])
    session: SessionCompactionConfig = Field(default_factory=SessionCompactionConfig)
    trajectory: TrajectoryCompactionConfig = Field(default_factory=TrajectoryCompactionConfig)

    @field_validator("offloading")
    @classmethod
    def unique_offloading_destinations(cls, value: str | list[str]) -> str | list[str]:
        if isinstance(value, list) and len(value) != len(set(value)):
            raise ValueError("Offloading destinations must not be repeated")
        return value


class HarnessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    compaction: CompactionConfig = Field(default_factory=CompactionConfig)


def ensure_config(cwd: Path) -> Path:
    """Create workspace defaults if absent, preserving existing files verbatim."""
    path = workspace_directory(cwd) / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    defaults = HarnessConfig()
    session = defaults.compaction.session
    trajectory = defaults.compaction.trajectory
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(
                f'[reasoning]\neagerness = "{defaults.reasoning.eagerness}"\n\n'
                f'[compaction]\noffloading = {json.dumps(defaults.compaction.offloading)}\n\n'
                f"[compaction.session]\ntrigger_tokens = {session.trigger_tokens}\n"
                f"keep_last_turns = {session.keep_last_turns}\n"
                f"max_critic_loops = {session.max_critic_loops}\n\n"
                f"[compaction.trajectory]\ntrigger_every_turns = {trajectory.trigger_every_turns}\n"
            )
    except FileExistsError:
        pass
    return path


def load_config(cwd: Path) -> HarnessConfig:
    """Read optional settings from the nearest workspace, without creating files."""
    path = workspace_directory(cwd) / "config.toml"
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return HarnessConfig()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"Cannot read harness configuration {path}: {error}") from error
    try:
        return HarnessConfig.model_validate(tomllib.loads(content))
    except ValueError as error:
        raise ValueError(f"Invalid harness configuration {path}: {error}") from error
