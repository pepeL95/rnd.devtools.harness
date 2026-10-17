from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BankPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Conservative starting points, not probabilities; see the retrieval calibration eval.
    automatic_min_similarity: float = Field(default=0.72, ge=0, le=1)
    search_min_similarity: float = Field(default=0.35, ge=0, le=1)
    automatic_results: int = Field(default=2, ge=1, le=5)
    search_results: int = Field(default=5, ge=1, le=20)
    automatic_tokens: int = Field(default=1200, ge=100, le=4000)
    search_tokens: int = Field(default=3000, ge=100, le=10000)

    @model_validator(mode="after")
    def thresholds(self) -> BankPolicy:
        if self.automatic_min_similarity < self.search_min_similarity:
            raise ValueError("automatic threshold must be at least the explicit-search threshold")
        return self


def load_policy(root: Path) -> BankPolicy:
    path = root / "config.toml"
    try:
        return BankPolicy.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return BankPolicy()
