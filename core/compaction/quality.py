from __future__ import annotations

import re
from dataclasses import dataclass


REQUIRED_HEADINGS = [
    "TASK HISTORY",
    "FAILED APPROACHES",
    "OPEN PROBLEMS",
    "IMPLICIT TASKS DISCOVERED",
    "NEXT STEPS",
    "TASK-APPROACH PAIRS",
    "GENERALIZABLE INSIGHTS",
    "SESSION NARRATIVE",
]

LEGACY_HEADINGS = [
    "ORIGINAL TASK",
    "USER DIRECTIVES",
]


@dataclass(frozen=True)
class QualityIssue:
    rule: str
    location: str
    fix: str


def quality_issues(memory_document: str) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    if not _has_markdown_document_structure(memory_document):
        issues.append(
            QualityIssue(
                rule="missing_markdown_hierarchy",
                location="document",
                fix=(
                    "Format the full memory document as markdown with coherent heading hierarchy such as "
                    "`##` for major sections and `###`/`####` for child structure."
                ),
            )
        )

    for heading in REQUIRED_HEADINGS:
        if not _has_heading(memory_document, heading):
            issues.append(
                QualityIssue(
                    rule="missing_required_heading",
                    location=heading,
                    fix=f"Add the `{heading}` section in the required order.",
                )
            )

    if _has_legacy_heading(memory_document):
        issues.append(
            QualityIssue(
                rule="legacy_heading_shape",
                location="TASK HISTORY",
                fix=(
                    "Replace transcript-era headings such as `ORIGINAL TASK` or `USER DIRECTIVES` with "
                    "semantic task markdown subsections under `TASK HISTORY`."
                ),
            )
        )

    if _has_compaction_artifacts(memory_document):
        issues.append(
            QualityIssue(
                rule="compaction_process_artifacts",
                location="document",
                fix=(
                    "Remove critic reasoning, reviser reasoning, revision logs, critique summaries, and any other "
                    "compaction-process artifacts. The final memory must contain session-related semantics only."
                ),
            )
        )

    if not _has_task_markdown_sections(memory_document):
        issues.append(
            QualityIssue(
                rule="missing_task_markdown_sections",
                location="TASK HISTORY",
                fix=(
                    "Structure each semantic task as its own markdown subsection using `#### [task title]` "
                    "with FULL-FIDELITY REF, TASK TIMESTAMPS, TASK DESCRIPTION SYNTHESIS, and EXECUTION MEMORY."
                ),
            )
        )
    elif _task_memories_too_thin(memory_document):
        issues.append(
            QualityIssue(
                rule="thin_task_memories",
                location="TASK HISTORY",
                fix=(
                    "Make each semantic task subsection read like a cohesive episodic memory: include a holistic task description, "
                    "a rich prose execution memory with meaningful findings and results, a full-fidelity reference, and task timestamps."
                ),
            )
        )

    if _missing_resume_signal(memory_document):
        issues.append(
            QualityIssue(
                rule="insufficient_resume_signal",
                location="TASK HISTORY / OPEN PROBLEMS / NEXT STEPS",
                fix=(
                    "Make the resumable state sharper through richer task history plus clearer open problems and next steps."
                ),
            )
        )

    if _missing_transfer_signal(memory_document):
        issues.append(
            QualityIssue(
                rule="insufficient_transfer_signal",
                location="TASK-APPROACH PAIRS / GENERALIZABLE INSIGHTS",
                fix=(
                    "Increase epistemic transfer by capturing durable mechanisms and "
                    "reusable lessons rather than chronology."
                ),
            )
        )

    return issues


def quality_report(memory_document: str) -> str | None:
    issues = quality_issues(memory_document)
    if not issues:
        return None
    lines = [
        "LOCAL QUALITY REVIEW",
        "These issues were detected by the compaction engine before approval.",
    ]
    for index, issue in enumerate(issues, start=1):
        lines.extend(
            [
                f"- RULE: {issue.rule}",
                f"  LOCATION: {issue.location}",
                f"  FIX: {issue.fix}",
            ]
        )
    lines.extend(
        [
            "",
            "CRITIQUE SUMMARY",
            f"  VIOLATIONS FOUND: {len(issues)}",
            "  SEVERITY: moderate",
            "  RECOMMENDED ACTION: revise targeted sections",
        ]
    )
    return "\n".join(lines)


def _has_heading(memory_document: str, heading: str) -> bool:
    pattern = rf"(?im)^\s*(?:#+\s*)?{re.escape(heading)}\b"
    return re.search(pattern, memory_document) is not None


def _has_legacy_heading(memory_document: str) -> bool:
    return any(_has_heading(memory_document, heading) for heading in LEGACY_HEADINGS)


def _has_task_markdown_sections(memory_document: str) -> bool:
    body = _section_body(memory_document, "TASK HISTORY")
    if body is None:
        return False
    return (
        "#### " in body
        and "- FULL-FIDELITY REF:" in body
        and "- TASK TIMESTAMPS:" in body
        and "- TASK DESCRIPTION SYNTHESIS:" in body
        and "- EXECUTION MEMORY:" in body
    )


def _task_memories_too_thin(memory_document: str) -> bool:
    body = _section_body(memory_document, "TASK HISTORY")
    if body is None:
        return False
    sections = [section.strip() for section in re.split(r"(?m)^####\s+", body) if section.strip()]
    if not sections:
        return True
    for section in sections:
        required_fields = [
            "- FULL-FIDELITY REF:",
            "- TASK TIMESTAMPS:",
            "- TASK DESCRIPTION SYNTHESIS:",
            "- EXECUTION MEMORY:",
        ]
        if any(field not in section for field in required_fields):
            return True
        description_line = _field_value(section, "TASK DESCRIPTION SYNTHESIS")
        execution_line = _field_value(section, "EXECUTION MEMORY")
        if description_line is None or len(re.findall(r"\b\w+\b", description_line)) < 12:
            return True
        if execution_line is None or len(re.findall(r"\b\w+\b", execution_line)) < 26:
            return True
        timestamp_line = _field_value(section, "TASK TIMESTAMPS")
        if timestamp_line is None or "->" not in timestamp_line:
            return True
    return False


def _has_markdown_document_structure(memory_document: str) -> bool:
    return "## " in memory_document and "### " in memory_document


def _missing_resume_signal(memory_document: str) -> bool:
    task_memories = _section_body(memory_document, "TASK HISTORY") or ""
    open_problems = _section_body(memory_document, "OPEN PROBLEMS") or ""
    next_steps = _section_body(memory_document, "NEXT STEPS") or ""
    return not task_memories.strip() or not open_problems.strip() or not next_steps.strip()


def _missing_transfer_signal(memory_document: str) -> bool:
    task_pairs = _section_body(memory_document, "TASK-APPROACH PAIRS") or ""
    insights = _section_body(memory_document, "GENERALIZABLE INSIGHTS") or ""
    return not task_pairs.strip() or not insights.strip()


def _section_body(memory_document: str, heading: str) -> str | None:
    headings = "|".join(re.escape(item) for item in REQUIRED_HEADINGS + LEGACY_HEADINGS)
    pattern = re.compile(
        rf"(?ims)^\s*(?:#+\s*)?{re.escape(heading)}\b\s*(.*?)"
        rf"(?=^\s*(?:#+\s*)?(?:{headings})\b|\Z)"
    )
    match = pattern.search(memory_document)
    if not match:
        return None
    return match.group(1).strip()


def _field_value(section: str, field: str) -> str | None:
    match = re.search(rf"(?im)^-\s+{re.escape(field)}:\s*(.+)$", section)
    if not match:
        return None
    return match.group(1).strip()


def sanitize_memory_document(memory_document: str) -> str:
    text = memory_document.strip()
    text = _drop_before_first_heading(text)
    text = _drop_revision_log(text)
    text = _drop_critique_summary(text)
    text = _drop_escalation(text)
    return text.strip()


def _has_compaction_artifacts(memory_document: str) -> bool:
    normalized = memory_document.upper()
    artifact_markers = [
        "REVISION LOG",
        "CRITIQUE SUMMARY",
        "LOCAL QUALITY REVIEW",
        "ESCALATION:",
        "MY THOUGHT PROCESS",
        "'TYPE': 'THINKING'",
        '"TYPE": "THINKING"',
        "**MY THOUGHT PROCESS",
    ]
    return any(marker in normalized for marker in artifact_markers)


def _drop_before_first_heading(text: str) -> str:
    match = re.search(r"(?m)^##\s+", text)
    if not match:
        return text
    return text[match.start():]


def _drop_revision_log(text: str) -> str:
    return re.sub(r"(?ims)^\s*(?:#+\s*)?REVISION LOG\b.*?(?=^\s*##\s|\Z)", "", text).strip()


def _drop_critique_summary(text: str) -> str:
    return re.sub(r"(?ims)^\s*CRITIQUE SUMMARY\b.*$", "", text).strip()


def _drop_escalation(text: str) -> str:
    return re.sub(r"(?im)^\s*ESCALATION:.*$", "", text).strip()
