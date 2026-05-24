from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.compaction.compactor import Compactor
from core.compaction.policy import CompactionPolicy
from core.compaction.serialization import events_to_trajectory
from core.session.events import EventType, SessionEvent


SECTION_HEADINGS = [
    "TASK REQUIREMENT SYNTHESIS",
    "CURRENT STATE",
    "WORK COMPLETED",
    "FAILED APPROACHES",
    "OPEN PROBLEMS",
    "IMPLICIT TASKS DISCOVERED",
    "NEXT STEPS",
    "CODEBASE CHARACTERISTICS",
    "TASK-APPROACH PAIRS",
    "GENERALIZABLE INSIGHTS",
    "SESSION NARRATIVE",
]


@dataclass(frozen=True)
class AtomScore:
    id: str
    label: str
    passed: bool
    matched: str | None


@dataclass(frozen=True)
class CaseScore:
    id: str
    title: str
    atom_score: float
    section_score: float
    semantic_richness_score: float
    passed_atoms: int
    total_atoms: int
    present_sections: int
    total_sections: int
    revisions: int
    compacted_event_count: int
    retained_event_count: int
    atom_details: list[AtomScore]
    missing_sections: list[str]


def main() -> None:
    args = _parse_args()
    fixture_path = args.fixture
    output_dir = _make_output_dir(args.output_root)

    fixture = _read_json(fixture_path)
    case_outputs: list[dict[str, Any]] = []
    scores: list[CaseScore] = []

    cases = _selected_cases(fixture["cases"], set(args.case_id))
    for case in cases:
        events = [_event_from_dict(item) for item in case["events"]]
        try:
            result = Compactor(
                policy=CompactionPolicy(
                    trigger_tokens=1,
                    keep_last_turns=int(case.get("keep_last_turns", args.keep_last_turns)),
                    max_critic_loops=args.critic_loops,
                )
            ).compact(events, token_estimate=999_999)
            restored_context = "\n\n".join(
                [result.memory_document, events_to_trajectory(result.events)]
            )
            score = _score_case(case, result.memory_document, restored_context, result)
            scores.append(score)
            case_output = {
                "case": {"id": case["id"], "title": case["title"]},
                "score": _as_json(score),
                "segmentation": result.segmentation,
                "memory_document": result.memory_document,
                "critiques": [critique.text for critique in result.critiques],
                "compacted_event_count": result.compacted_event_count,
                "retained_event_count": result.retained_event_count,
            }
            _write_text(output_dir / f"{case['id']}.md", _case_markdown(case_output))
        except Exception as exc:  # noqa: BLE001 - eval output must preserve provider failures.
            case_output = {
                "case": {"id": case["id"], "title": case["title"]},
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        case_outputs.append(case_output)

    summary = _summary(fixture, scores, case_outputs, output_dir)
    _write_json(output_dir / "results.json", summary)
    _write_text(output_dir / "summary.md", _summary_markdown(summary))
    print(output_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compaction semantic-richness evaluations.")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("evals/compaction_semantic_richness_set.json"),
        help="Path to the semantic-richness fixture set.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("eval_results/compaction_semantic_richness"),
        help="Directory where timestamped eval outputs are written.",
    )
    parser.add_argument(
        "--critic-loops",
        type=int,
        default=1,
        help="Number of critic/revision loops to run per case.",
    )
    parser.add_argument(
        "--keep-last-turns",
        type=int,
        default=2,
        help="Fallback retained-turn count when a case does not specify one.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Run only the selected case id. May be provided more than once.",
    )
    return parser.parse_args()


def _score_case(case: dict[str, Any], memory_document: str, restored_context: str, result: Any) -> CaseScore:
    atom_details = [_score_atom(atom, restored_context) for atom in case["expected_atoms"]]
    present_sections = [heading for heading in SECTION_HEADINGS if _has_heading(memory_document, heading)]
    missing_sections = [heading for heading in SECTION_HEADINGS if heading not in present_sections]
    passed_atoms = sum(1 for atom in atom_details if atom.passed)
    atom_score = passed_atoms / len(atom_details)
    section_score = len(present_sections) / len(SECTION_HEADINGS)
    semantic_richness_score = round((atom_score * 0.75) + (section_score * 0.25), 4)
    return CaseScore(
        id=str(case["id"]),
        title=str(case["title"]),
        atom_score=round(atom_score, 4),
        section_score=round(section_score, 4),
        semantic_richness_score=semantic_richness_score,
        passed_atoms=passed_atoms,
        total_atoms=len(atom_details),
        present_sections=len(present_sections),
        total_sections=len(SECTION_HEADINGS),
        revisions=result.revisions,
        compacted_event_count=result.compacted_event_count,
        retained_event_count=result.retained_event_count,
        atom_details=atom_details,
        missing_sections=missing_sections,
    )


def _score_atom(atom: dict[str, Any], restored_context: str) -> AtomScore:
    haystack = _normalize(restored_context)
    for needle in atom["must_include_any"]:
        if _normalize(str(needle)) in haystack:
            return AtomScore(
                id=str(atom["id"]),
                label=str(atom["label"]),
                passed=True,
                matched=str(needle),
            )
    return AtomScore(
        id=str(atom["id"]),
        label=str(atom["label"]),
        passed=False,
        matched=None,
    )


def _selected_cases(cases: list[dict[str, Any]], case_ids: set[str]) -> list[dict[str, Any]]:
    if not case_ids:
        return cases
    selected = [case for case in cases if str(case["id"]) in case_ids]
    missing = sorted(case_ids - {str(case["id"]) for case in selected})
    if missing:
        raise ValueError(f"Unknown case id(s): {', '.join(missing)}")
    return selected


def _has_heading(memory_document: str, heading: str) -> bool:
    pattern = rf"(?im)^\s*(?:#+\s*)?{re.escape(heading)}\b"
    return re.search(pattern, memory_document) is not None


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def _event_from_dict(data: dict[str, Any]) -> SessionEvent:
    return SessionEvent(
        type=EventType(str(data["type"])),
        turn=int(data["turn"]),
        timestamp=str(data["timestamp"]),
        payload=dict(data["payload"]),
    )


def _make_output_dir(root: Path) -> Path:
    output_dir = root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _summary(
    fixture: dict[str, Any],
    scores: list[CaseScore],
    case_outputs: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    errors = [item for item in case_outputs if "error" in item]
    average = (
        round(sum(score.semantic_richness_score for score in scores) / len(scores), 4)
        if scores
        else 0.0
    )
    return {
        "fixture": {
            "name": fixture["name"],
            "version": fixture["version"],
            "case_count": len(case_outputs),
        },
        "run": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(output_dir),
            "scored_case_count": len(scores),
            "error_count": len(errors),
        },
        "aggregate": {
            "average_semantic_richness_score": average,
        },
        "scores": [_as_json(score) for score in scores],
        "errors": errors,
    }


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Compaction Semantic Richness Eval",
        "",
        f"Output directory: `{summary['run']['output_dir']}`",
        f"Cases scored: {summary['run']['scored_case_count']} / {summary['fixture']['case_count']}",
        f"Errors: {summary['run']['error_count']}",
        f"Average semantic richness score: {summary['aggregate']['average_semantic_richness_score']}",
        "",
        "| Case | Score | Atoms | Sections | Revisions |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for score in summary["scores"]:
        lines.append(
            "| {id} | {semantic_richness_score:.4f} | {passed_atoms}/{total_atoms} | "
            "{present_sections}/{total_sections} | {revisions} |".format(**score)
        )
    if summary["errors"]:
        lines.extend(["", "## Errors", ""])
        for item in summary["errors"]:
            lines.append(
                f"- `{item['case']['id']}`: {item['error']['type']}: {item['error']['message']}"
            )
    return "\n".join(lines) + "\n"


def _case_markdown(case_output: dict[str, Any]) -> str:
    score = case_output["score"]
    lines = [
        f"# {case_output['case']['title']}",
        "",
        f"Case: `{case_output['case']['id']}`",
        f"Semantic richness score: {score['semantic_richness_score']}",
        f"Atoms: {score['passed_atoms']} / {score['total_atoms']}",
        f"Sections: {score['present_sections']} / {score['total_sections']}",
        f"Revisions: {score['revisions']}",
        "",
        "## Missing Atoms",
        "",
    ]
    missing = [atom for atom in score["atom_details"] if not atom["passed"]]
    if missing:
        lines.extend(f"- {atom['id']}: {atom['label']}" for atom in missing)
    else:
        lines.append("- None")
    lines.extend(["", "## Missing Sections", ""])
    if score["missing_sections"]:
        lines.extend(f"- {section}" for section in score["missing_sections"])
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Segmentation",
            "",
            case_output["segmentation"],
            "",
            "## Memory Document",
            "",
            case_output["memory_document"],
            "",
            "## Critiques",
            "",
        ]
    )
    if case_output["critiques"]:
        for index, critique in enumerate(case_output["critiques"], start=1):
            lines.extend([f"### Critique {index}", "", critique, ""])
    else:
        lines.append("- None")
    return "\n".join(lines).strip() + "\n"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _write_text(path: Path, content: str) -> None:
    path.write_text(content)


def _as_json(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
