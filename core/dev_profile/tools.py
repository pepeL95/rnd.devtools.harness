from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from core.dev_profile.store import DevProfileConflictError, DevProfileStore
from core.session.events import EventType, SessionEvent

EMPTY_DEV_PROFILE = "# Developer Preferences\n\n## Status\n\nNo dev preferences learned yet."


class InspectSessionSchema(BaseModel):
    start_turn: int = Field(default=1, ge=1, description="First completed turn to include.")
    limit: int = Field(default=20, ge=1, le=50, description="Maximum number of turn summaries to return.")


class ReadSessionTurnsSchema(BaseModel):
    turns: list[int] = Field(description="One to five completed turn numbers to read in full.", min_length=1, max_length=5)


class SearchSessionSchema(BaseModel):
    query: str = Field(description="Case-insensitive text to find in the dumped session.", min_length=1)
    limit: int = Field(default=20, ge=1, le=50, description="Maximum matching events to return.")


class DevProfileEvidence(BaseModel):
    turn: int = Field(ge=1, description="Completed turn containing the user-authored evidence.")
    quote: str = Field(min_length=1, description="Exact quote from a user event in that turn.")


class UpdateDevProfileSchema(BaseModel):
    content: str = Field(description="Complete free-form Markdown content for DEVPROFILE.md.")
    expected_revision: str | None = Field(
        default=None,
        description="Revision returned by read_devprofile, or null when the file did not exist.",
    )
    evidence: list[DevProfileEvidence] = Field(
        default_factory=list,
        description="Exact user-authored quotes supporting a substantive profile update.",
    )


def create_dev_profile_tools(
    events: tuple[SessionEvent, ...],
    store: DevProfileStore,
) -> list[BaseTool]:
    """Create tools scoped to an immutable completed-session snapshot."""

    by_turn: dict[int, list[SessionEvent]] = {}
    for event in events:
        by_turn.setdefault(event.turn, []).append(event)

    def inspect_session(start_turn: int = 1, limit: int = 20) -> str:
        """List compact previews of completed turns before choosing what to inspect."""

        selected_turns = [turn for turn in sorted(by_turn) if turn >= start_turn][:limit]
        items = []
        for turn in selected_turns:
            turn_events = by_turn[turn]
            counts = Counter(event.type.value for event in turn_events)
            items.append(
                {
                    "turn": turn,
                    "user_messages": _event_contents(turn_events, EventType.USER),
                    "event_counts": dict(sorted(counts.items())),
                }
            )
        remaining = [turn for turn in sorted(by_turn) if turn > (selected_turns[-1] if selected_turns else start_turn - 1)]
        return json.dumps(
            {
                "turns": items,
                "next_start_turn": remaining[0] if remaining else None,
                "total_completed_turns": len(by_turn),
            },
            ensure_ascii=False,
        )

    def read_session_turns(turns: list[int]) -> str:
        """Read selected completed turns from the full-fidelity dumped session."""

        payload = {
            str(turn): [_event_with_evidence_role(event) for event in by_turn.get(turn, [])]
            for turn in turns
        }
        return _bounded_json(payload)

    def search_session(query: str, limit: int = 20) -> str:
        """Search the dumped session for exact developer language or repeated evidence."""

        needle = query.casefold()
        matches: list[dict[str, Any]] = []
        for event in events:
            if event.type != EventType.USER:
                continue
            serialized = json.dumps(event.payload, ensure_ascii=False)
            if needle not in serialized.casefold():
                continue
            matches.append(event.to_json_dict())
            if len(matches) >= limit:
                break
        return _bounded_json({"query": query, "matches": matches})

    def read_devprofile() -> str:
        """Read the current free-form developer profile and its revision."""

        snapshot = store.read()
        return json.dumps(
            {
                "path": str(snapshot.path),
                "exists": snapshot.exists,
                "revision": snapshot.revision,
                "content": snapshot.content,
            },
            ensure_ascii=False,
        )

    def update_devprofile(
        content: str,
        expected_revision: str | None = None,
        evidence: list[DevProfileEvidence] | None = None,
    ) -> str:
        """Create or atomically replace DEVPROFILE.md after verifying its revision."""

        references = evidence or []
        if not _has_heading_hierarchy(content):
            return json.dumps(
                {
                    "status": "rejected",
                    "message": "DEVPROFILE.md must use GitHub-flavored Markdown with an ATX heading and subheading.",
                }
            )
        current = store.read()
        if (
            content.strip() == EMPTY_DEV_PROFILE
            and current.exists
            and current.content.strip()
            and current.content.strip() != EMPTY_DEV_PROFILE
        ):
            return json.dumps(
                {
                    "status": "rejected",
                    "message": "A populated DEVPROFILE.md cannot be replaced with the empty-state document.",
                }
            )
        if content.strip() != EMPTY_DEV_PROFILE:
            if not references:
                return json.dumps(
                    {
                        "status": "rejected",
                        "message": "Substantive profile updates require exact quotes from user events.",
                    }
                )
            invalid = [reference.model_dump() for reference in references if not _valid_user_evidence(reference, by_turn)]
            if invalid:
                return json.dumps(
                    {
                        "status": "rejected",
                        "message": "One or more evidence quotes were not found in user events.",
                        "invalid_evidence": invalid,
                    },
                    ensure_ascii=False,
                )
        try:
            updated = store.update(content, expected_revision=expected_revision)
        except DevProfileConflictError as exc:
            return json.dumps(
                {
                    "status": "conflict",
                    "current_revision": exc.current_revision,
                    "message": str(exc),
                }
            )
        return json.dumps(
            {
                "status": "updated",
                "path": str(updated.path),
                "revision": updated.revision,
            }
        )

    return [
        StructuredTool.from_function(
            func=inspect_session,
            name="inspect_session",
            description="List user-authored messages from completed turns. Use this first to identify possible developer preferences without being biased by assistant behavior.",
            args_schema=InspectSessionSchema,
            infer_schema=False,
        ),
        StructuredTool.from_function(
            func=read_session_turns,
            name="read_session_turns",
            description="Read one to five completed turns from the full dump. User events are preference evidence; all assistant, reasoning, and tool events are context only.",
            args_schema=ReadSessionTurnsSchema,
            infer_schema=False,
        ),
        StructuredTool.from_function(
            func=search_session,
            name="search_session",
            description="Search only user-authored events for exact language or repeated preference evidence.",
            args_schema=SearchSessionSchema,
            infer_schema=False,
        ),
        StructuredTool.from_function(
            func=read_devprofile,
            name="read_devprofile",
            description="Read the current free-form DEVPROFILE.md and its revision. Always call this before updating the profile.",
        ),
        StructuredTool.from_function(
            func=update_devprofile,
            name="update_devprofile",
            description=f"Create or atomically replace the complete merged DEVPROFILE.md as GitHub-flavored Markdown with descriptive ATX headings and subheadings. Preserve existing instructions unless newer user evidence explicitly changes them. The heading names and document structure are flexible. Substantive content requires exact user quotes; when the file does not exist and no supported preference exists, write exactly: {EMPTY_DEV_PROFILE}",
            args_schema=UpdateDevProfileSchema,
            infer_schema=False,
        ),
    ]


def completed_dump_snapshot(events: list[SessionEvent]) -> tuple[SessionEvent, ...]:
    completed_turns = {event.turn for event in events if event.type == EventType.TURN_END}
    return tuple(event for event in events if event.turn in completed_turns and event.turn > 0)


def _event_contents(events: list[SessionEvent], event_type: EventType) -> list[str]:
    return [
        " ".join(str(event.payload.get("content") or "").split())[:300]
        for event in events
        if event.type == event_type and str(event.payload.get("content") or "").strip()
    ]


def _event_with_evidence_role(event: SessionEvent) -> dict[str, Any]:
    serialized = event.to_json_dict()
    serialized["preference_evidence"] = event.type == EventType.USER
    return serialized


def _valid_user_evidence(reference: DevProfileEvidence, by_turn: dict[int, list[SessionEvent]]) -> bool:
    quote = reference.quote.strip()
    if not quote:
        return False
    return any(
        quote in str(event.payload.get("content") or "")
        for event in by_turn.get(reference.turn, [])
        if event.type == EventType.USER
    )


def _has_heading_hierarchy(content: str) -> bool:
    return bool(re.search(r"^# [^#\n].+$", content, re.MULTILINE)) and bool(
        re.search(r"^## [^#\n].+$", content, re.MULTILINE)
    )


def _bounded_json(value: Any, *, max_chars: int = 60000) -> str:
    rendered = json.dumps(value, ensure_ascii=False)
    if len(rendered) <= max_chars:
        return rendered
    return json.dumps(
        {"truncated": True, "content_prefix": rendered[:max_chars]},
        ensure_ascii=False,
    )
