from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from core.utilities.workspace import workspace_directory


ReasoningEagerness = Literal["low", "medium", "high"]


class ReasoningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    eagerness: ReasoningEagerness = "high"


class HarnessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)


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
