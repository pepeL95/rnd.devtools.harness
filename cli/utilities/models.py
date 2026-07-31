from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSummary:
    name: str
    description: str | None = None
    is_current: bool = False


def default_models(*, current_model: str | None = None) -> list[ModelSummary]:
    models = [
        ModelSummary("gemini-3.1-flash-lite", "Default low-latency driver model"),
        ModelSummary("gemini-2.5-pro", "Higher-capability Gemini model"),
    ]
    if current_model and all(summary.name != current_model for summary in models):
        models.insert(0, ModelSummary(current_model, "Current environment model"))
    if current_model is None:
        return models
    return [
        ModelSummary(summary.name, summary.description, is_current=summary.name == current_model)
        for summary in models
    ]
