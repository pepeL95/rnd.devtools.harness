from __future__ import annotations

import json
import logging
from hashlib import sha256
from typing import Any, TYPE_CHECKING

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from core.compaction.models import ArcUpdate, CompactionResult, Critique, TaskArc
from core.compaction.policy import CompactionPolicy
from core.compaction.prompts import ARC_PROMPT, ARC_REVIEW_PROMPT
from core.compaction.serialization import events_to_trajectory, memory_restore_message, render_arc_memory
from core.compaction.token_counter import TokenCounter
from core.compaction.window import split_compaction_window
from core.session.events import EventType, SessionEvent
from core.utilities.defaults import get_model_name

if TYPE_CHECKING:
    from core.session.manager import SessionManager
    from core.trajectory_bank.store import TrajectoryBank


class Compactor:
    """Maintain cumulative task arcs, optionally archiving evicted trajectory evidence."""

    def __init__(
        self,
        policy: CompactionPolicy | None = None,
        token_counter: TokenCounter | None = None,
        *,
        trajectory_bank: TrajectoryBank | None = None,
        session_manager: SessionManager | None = None,
    ) -> None:
        self.policy = policy or CompactionPolicy()
        self.policy.validate()
        self.token_counter = token_counter or TokenCounter()
        if trajectory_bank is not None and session_manager is None:
            raise ValueError("session_manager is required when trajectory_bank is supplied")
        self.trajectory_bank = trajectory_bank
        self.session_manager = session_manager

    def compact(
        self,
        events: list[SessionEvent],
        token_estimate: int | None = None,
    ) -> CompactionResult:
        bank = self.trajectory_bank
        manager = self.session_manager
        window = split_compaction_window(events, self.policy.keep_last_turns)
        usage = _TokenUsage()
        previous: dict[str, TaskArc] = {}
        manifest: dict[str, list[str]] = {}
        new_events = []
        prior_context = []
        for event in window.compacted:
            if event.payload.get("kind") == "memory_restore":
                prior_context.append(
                    {key: event.payload[key] for key in ("narrative", "current_state")}
                    if "narrative" in event.payload and "current_state" in event.payload
                    else str(event.payload.get("content", ""))
                )
                for item in event.payload.get("task_arcs", []):
                    arc = TaskArc.model_validate(item)
                    previous[arc.arc_id] = arc
                manifest.update(event.payload.get("trajectory_manifest", {}))
                if not event.payload.get("task_arcs"):
                    # Preserve legacy memory verbatim rather than manufacture missing source references.
                    id = "legacy-" + sha256(str(event.payload.get("content", "")).encode()).hexdigest()[:16]
                    previous[id] = TaskArc(arc_id=id, title="Prior session memory", status="active",
                                          intent="Preserve pre-bank task history", implementation="Legacy compacted context",
                                          findings=str(event.payload.get("content", "")), constraints="See preserved memory",
                                          validation="No bank references available for this legacy summary", remaining="See preserved memory")
            else:
                new_events.append(event)
        meaningful = {EventType.USER, EventType.ASSISTANT, EventType.REASONING, EventType.TOOL, EventType.TOOL_OUTPUT}
        turns = sorted({event.turn for event in new_events if event.type in meaningful})
        if not turns:
            return CompactionResult(events=list(events), memory_document="", segmentation="",
                                    retained_event_count=len(events), token_usage=usage.as_dict(),
                                    token_estimate_before=token_estimate if token_estimate is not None
                                    else self.token_counter.count_events(events))
        prior_ids = {id for ids in manifest.values() for id in ids}
        if bank is not None and prior_ids and {record.id for record in bank.get(ids=sorted(prior_ids))} != prior_ids:
            raise ValueError("Previously archived trajectory references are missing; refusing to compact")

        schema = ArcUpdate.model_json_schema()
        schema["$defs"]["TaskArc"]["properties"]["arc_id"]["enum"] = ["", *previous]
        prompt = json.dumps({
            "schema": schema, "existing_arcs": [arc.model_dump() for arc in previous.values()],
            "previous_session_context": prior_context,
            "new_turns": turns, "trajectory": events_to_trajectory(new_events),
            "retained_tail_for_orientation_only": events_to_trajectory(window.retained),
        }, ensure_ascii=False)
        update, raw, revisions = _synthesize_update(self, prompt, turns, previous, usage)
        critiques = []
        model_names = {"compactor": get_model_name(self.policy.compactor_model)}
        if self.policy.max_critic_loops:
            model_names["critic"] = get_model_name(self.policy.critic_model)
            for review_index in range(self.policy.max_critic_loops + 1):
                review = _invoke_text(self.policy.critic_model, ARC_REVIEW_PROMPT,
                                      prompt + "\nProposed update:\n" + update.model_dump_json(),
                                      token_counter=self.token_counter, usage=usage).strip()
                approved = review == "APPROVE"
                critiques.append(Critique(review))
                if approved:
                    break
                if review_index == self.policy.max_critic_loops:
                    logging.getLogger(__name__).warning("Task arc review budget exhausted; using the latest revision.")
                    break
                update, raw, repairs = _synthesize_update(
                    self, prompt + "\nPrevious update:\n" + update.model_dump_json()
                    + "\nRequired fidelity corrections:\n" + review, turns, previous, usage,
                )
                revisions += 1 + repairs

        all_arcs = dict(previous)
        updated_arcs = []
        for arc in update.arcs:
            source_id = next((event.id for event in new_events if event.turn in arc.turns), "")
            id = arc.arc_id or "arc-" + sha256(source_id.encode()).hexdigest()[:20]
            prior = previous.get(id)
            grounded = events_to_trajectory([event for event in new_events if event.turn in arc.turns]).lower()
            keywords = list(dict.fromkeys(word for word in arc.keywords if word.strip() and word.lower() in grounded))
            arc = arc.model_copy(update={"arc_id": id, "turns": sorted(set(arc.turns + (prior.turns if prior else []))),
                                          "keywords": list(dict.fromkeys((prior.keywords if prior else []) + keywords))[:12]})
            all_arcs[id] = arc
            updated_arcs.append(arc)

        if bank is not None:
            from core.trajectory_bank.archival import archive_trajectories

            archive_trajectories(bank, manager, updated_arcs, new_events, turns, manifest)
        memory = render_arc_memory(update.narrative, update.current_state, list(all_arcs.values()),
                                   manager.session_id if manager else "", manifest)
        memory_event = SessionEvent(type=EventType.USER, turn=window.compacted[0].turn, payload={
            "role": "user", "kind": "memory_restore", "content": memory_restore_message(memory),
            "task_arcs": [arc.model_dump() for arc in all_arcs.values()], "trajectory_manifest": manifest,
            "narrative": update.narrative, "current_state": update.current_state,
            "compacted_event_count": len(window.compacted), "retained_event_count": len(window.retained),
        })
        return CompactionResult(events=[memory_event, *window.retained], memory_document=memory, segmentation=raw,
                                revisions=revisions, critiques=critiques, model_names=model_names, token_usage=usage.as_dict(),
                                token_estimate_before=token_estimate if token_estimate is not None
                                else self.token_counter.count_events(events),
                                compacted_event_count=len(window.compacted), retained_event_count=len(window.retained))


def _synthesize_update(
    compactor: Compactor, prompt: str, turns: list[int], previous: dict[str, TaskArc], usage: _TokenUsage,
) -> tuple[ArcUpdate, str, int]:
    error = ""
    for attempt in range(2):
        raw = _invoke_text(compactor.policy.compactor_model, ARC_PROMPT, prompt + error,
                           token_counter=compactor.token_counter, usage=usage)
        try:
            update = ArcUpdate.model_validate_json(raw)
            assigned = [turn for arc in update.arcs for turn in arc.turns]
            if sorted(assigned) != turns:
                raise ValueError("Every new turn must be assigned exactly once; no tail or invented turns allowed")
            ids = [arc.arc_id for arc in update.arcs if arc.arc_id]
            if len(ids) != len(set(ids)):
                raise ValueError("Existing arc IDs must be unique")
            unknown = [id for id in ids if id not in previous]
            if unknown:
                raise ValueError(f"Invented arc IDs {unknown!r}. Every new task MUST use the literal field "
                                 '\"arc_id\": \"\". The harness assigns IDs, not you.')
            if any(not arc.turns and not arc.arc_id for arc in update.arcs):
                raise ValueError("New arcs require source turns")
            return update, raw, attempt
        except ValueError as exc:
            if attempt == 1:
                raise
            error = f"\nPrevious output failed validation: {exc}. Return corrected JSON, not commentary."

    raise AssertionError("Unreachable synthesis retry state")


def _invoke_text(
    model: BaseChatModel,
    system_prompt: str,
    user_prompt: str,
    *,
    token_counter: TokenCounter,
    usage: "_TokenUsage",
) -> str:
    prompt = f"{system_prompt}\n\n{user_prompt}"
    usage.input_tokens += token_counter.count_text(prompt)
    usage.call_count += 1
    response = model.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    content = getattr(response, "content", response)
    text = _stringify_content(content)
    usage.output_tokens += token_counter.count_text(text)
    return text


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


class _TokenUsage:
    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.call_count = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "call_count": self.call_count,
        }
