from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any, TYPE_CHECKING

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from core.compaction.episodes import split_task_episodes
from core.compaction.models import CompactionResult, Critique
from core.compaction.offloading import BaseCompactionOffloadingStrategy, CompactionOffloadingStack
from core.compaction.policy import CompactionPolicy
from core.compaction.prompts import ARC_PROMPT, ARC_REVIEW_PROMPT, TASK_EPISODES_PROMPT
from core.compaction.serialization import events_to_trajectory, memory_restore_message
from core.compaction.token_counter import TokenCounter
from core.compaction.usage import TokenUsage as _TokenUsage
from core.compaction.window import split_compaction_window
from core.session.events import EventType, SessionEvent
from core.utilities.defaults import get_model_name

if TYPE_CHECKING:
    from core.session.manager import SessionManager


class Compactor:
    """Synthesize a cumulative session arc with optional evidence persistence."""

    def __init__(
        self,
        policy: CompactionPolicy | None = None,
        token_counter: TokenCounter | None = None,
        *,
        offloading: BaseCompactionOffloadingStrategy | Sequence[BaseCompactionOffloadingStrategy] | None = None,
        session_manager: SessionManager | None = None,
    ) -> None:
        self.policy = policy or CompactionPolicy()
        self.policy.validate()
        self.token_counter = token_counter or TokenCounter()
        self.offloading = (
            offloading if offloading is None or isinstance(offloading, BaseCompactionOffloadingStrategy)
            else CompactionOffloadingStack(offloading)
        )
        self.session_manager = session_manager

    def compact(
        self,
        events: list[SessionEvent],
        token_estimate: int | None = None,
    ) -> CompactionResult:
        window = split_compaction_window(events, self.policy.keep_last_turns)
        usage = _TokenUsage()
        previous = []
        manifest: dict[str, list[str]] = {}
        archive_references: list[str] = []
        new_events = []
        for event in window.compacted:
            if event.payload.get("kind") == "memory_restore":
                previous.append(event.payload.get("session_arc", event.payload.get("content", "")))
                archive_references.extend(event.payload.get("evidence_references", []))
                for group, ids in event.payload.get("trajectory_manifest", {}).items():
                    manifest[group] = list(dict.fromkeys([*manifest.get(group, []), *ids]))
            else:
                new_events.append(event)
        meaningful = {EventType.USER, EventType.ASSISTANT, EventType.REASONING, EventType.TOOL, EventType.TOOL_OUTPUT}
        turns = sorted({event.turn for event in new_events if event.type in meaningful})
        estimated = token_estimate if token_estimate is not None else self.token_counter.count_events(events)
        if not turns:
            return CompactionResult(events=list(events), memory_document="", segmentation="",
                                    retained_event_count=len(events), token_usage=usage.as_dict(),
                                    token_estimate_before=estimated)

        strategy, manager = self.offloading, self.session_manager
        if strategy is not None:
            strategy.validate(manifest)
        sources = {
            "previous_session_arc": previous,
            "trajectory": events_to_trajectory(new_events),
        }
        extraction = _invoke_text(
            self.policy.compactor_model, TASK_EPISODES_PROMPT, json.dumps(sources, ensure_ascii=False),
            token_counter=self.token_counter, usage=usage,
        ).strip()
        if not extraction:
            raise ValueError("Task episodes are empty; keeping original session context")
        episodes = split_task_episodes(extraction)
        evidence = strategy.prepare(new_events, turns, episodes, manifest) if strategy is not None else None
        if evidence is not None:
            manifest = evidence.manifest
            archive_references.extend(evidence.references)
        archive_references = list(dict.fromkeys(archive_references))
        references = list(archive_references)
        for group, ids in manifest.items():
            if manager:
                references.append(f"trajectory://{manager.session_id}/{group} ({len(ids)} trajectories)")
            else:
                references.extend(f"trajectory-record://{id}" for id in ids)
        if manager:
            references.append(f"Raw session: {manager.dump_path} (look up the cited turn numbers).")
        prompt = json.dumps({
            **sources,
            "task_episodes": [{"content": episode, "reference": evidence.episode_references[index] if evidence else ""}
                              for index, episode in enumerate(episodes)],
            "retained_tail_for_orientation_only": events_to_trajectory(window.retained),
            "evidence_references": references,
        }, ensure_ascii=False)
        memory = _invoke_text(self.policy.compactor_model, ARC_PROMPT, prompt,
                              token_counter=self.token_counter, usage=usage).strip()
        if not memory:
            raise ValueError("Session arc is empty; keeping original session context")
        critiques = []
        revisions = 0
        model_names = {"compactor": get_model_name(self.policy.compactor_model)}
        if self.policy.max_critic_loops:
            model_names["critic"] = get_model_name(self.policy.critic_model)
            for review_index in range(self.policy.max_critic_loops + 1):
                review = _invoke_text(self.policy.critic_model, ARC_REVIEW_PROMPT,
                                      prompt + "\nProposed session arc:\n" + memory,
                                      token_counter=self.token_counter, usage=usage).strip()
                critiques.append(Critique(review))
                if critiques[-1].approved:
                    break
                if review_index == self.policy.max_critic_loops:
                    logging.getLogger(__name__).warning("Session arc review budget exhausted; using the latest revision.")
                    break
                revised = _invoke_text(
                    self.policy.compactor_model, ARC_PROMPT,
                    prompt + "\nPrevious draft:\n" + memory + "\nFidelity corrections:\n" + review,
                    token_counter=self.token_counter, usage=usage,
                ).strip()
                revisions += 1
                if not revised:
                    logging.getLogger(__name__).warning("Session arc revision was empty; using the last nonempty draft.")
                    break
                memory = revised

        # References and publication are harness-owned, independent of the model's prose.
        document = memory
        if references:
            document += "\n\nEvidence references:\n" + "\n".join(references)
        memory_event = SessionEvent(type=EventType.USER, turn=window.compacted[0].turn, payload={
            "role": "user", "kind": "memory_restore", "content": memory_restore_message(document),
            "session_arc": memory, "trajectory_manifest": manifest,
            "evidence_references": archive_references,
            "compacted_event_count": len(window.compacted), "retained_event_count": len(window.retained),
        })
        result = CompactionResult(events=[memory_event, *window.retained], memory_document=document, segmentation=extraction,
                                revisions=revisions, critiques=critiques, model_names=model_names, token_usage=usage.as_dict(),
                                token_estimate_before=estimated, compacted_event_count=len(window.compacted),
                                retained_event_count=len(window.retained))
        if evidence is not None:
            evidence.persist(result)
        return result


def _invoke_text(
    model: BaseChatModel,
    system_prompt: str,
    user_prompt: str,
    *,
    token_counter: TokenCounter,
    usage: "_TokenUsage",
) -> str:
    prompt = f"{system_prompt}\n\n{user_prompt}"
    usage.call_count += 1
    response = model.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    content = getattr(response, "content", response)
    text = _stringify_content(content)
    usage.record(response, prompt, text, token_counter)
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
