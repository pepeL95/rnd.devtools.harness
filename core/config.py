from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from core.trajectory_bank.policy import BankPolicy
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

    session: SessionCompactionConfig = Field(default_factory=SessionCompactionConfig)
    trajectory: TrajectoryCompactionConfig = Field(default_factory=TrajectoryCompactionConfig)


class HarnessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    compaction: CompactionConfig = Field(default_factory=CompactionConfig)
    trajectory_bank: BankPolicy = Field(default_factory=BankPolicy)


def ensure_config(cwd: Path) -> Path:
    """Create defaults or append missing top-level sections without rewriting settings."""
    path = workspace_directory(cwd) / "config.toml"
    _reject_legacy_bank_config(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    defaults = HarnessConfig()
    session = defaults.compaction.session
    trajectory = defaults.compaction.trajectory
    bank = defaults.trajectory_bank
    sections = {
        "reasoning": f'[reasoning]\neagerness = "{defaults.reasoning.eagerness}"\n',
        "compaction": (
            f"[compaction.session]\ntrigger_tokens = {session.trigger_tokens}\n"
            f"keep_last_turns = {session.keep_last_turns}\n"
            f"max_critic_loops = {session.max_critic_loops}\n\n"
            f"[compaction.trajectory]\ntrigger_every_turns = {trajectory.trigger_every_turns}\n"
        ),
        "trajectory_bank": (
            f"[trajectory_bank]\nenabled = {str(bank.enabled).lower()}\n"
            f"automatic_min_similarity = {bank.automatic_min_similarity}\n"
            f"search_min_similarity = {bank.search_min_similarity}\n"
            f"automatic_results = {bank.automatic_results}\nsearch_results = {bank.search_results}\n"
            f"automatic_tokens = {bank.automatic_tokens}\nsearch_tokens = {bank.search_tokens}\n"
        ),
    }
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write("\n".join(sections.values()))
    except FileExistsError:
        # Validate before adding anything; malformed user settings must remain untouched.
        load_config(cwd)
        document = tomllib.loads(path.read_text(encoding="utf-8"))
        missing = [text for name, text in sections.items() if name not in document]
        if missing:
            with path.open("a", encoding="utf-8") as handle:
                handle.write("\n\n" + "\n".join(missing))
    return path


def load_config(cwd: Path) -> HarnessConfig:
    """Read optional settings from the nearest workspace, without creating files."""
    path = workspace_directory(cwd) / "config.toml"
    _reject_legacy_bank_config(path.parent)
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


def _reject_legacy_bank_config(workspace: Path) -> None:
    legacy = workspace / ".trajectorybank/config.toml"
    if legacy.exists():
        raise ValueError(
            f"Move settings from {legacy} into [trajectory_bank] in {workspace / 'config.toml'}, "
            "then remove the old config file. Leave the Chroma database untouched."
        )
