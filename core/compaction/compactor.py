from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

from core.compaction.llm import LangChainTextGenerator, TextGenerator
from core.compaction.models import CompactionResult, Critique
from core.compaction.policy import CompactionPolicy
from core.compaction.prompts import CRITIC_PROMPT, REVISION_PROMPT, SEGMENTATION_PROMPT, SYNTHESIS_PROMPT
from core.compaction.quality import quality_report, sanitize_memory_document
from core.compaction.serialization import events_to_trajectory, memory_restore_message
from core.compaction.token_counter import TokenCounter
from core.compaction.window import split_compaction_window
from core.session.events import EventType, SessionEvent


class Compactor:
    """Run the segmentation -> synthesis -> critic -> revision compaction loop."""

    def __init__(
        self,
        generator: TextGenerator | None = None,
        policy: CompactionPolicy | None = None,
        token_counter: TokenCounter | None = None,
        on_stage: Callable[[str, dict[str, int | str]], None] | None = None,
    ) -> None:
        self.policy = policy or CompactionPolicy()
        self.policy.validate()
        self.generator = generator or LangChainTextGenerator.from_chat_model(self.policy.model)
        self.token_counter = token_counter or TokenCounter()
        self.on_stage = on_stage

    def compact(
        self,
        events: list[SessionEvent],
        token_estimate: int | None = None,
    ) -> CompactionResult:
        window = split_compaction_window(events, self.policy.keep_last_turns)
        if not window.should_compact:
            return CompactionResult(
                events=list(events),
                memory_document="",
                segmentation="",
                token_estimate_before=token_estimate or self.token_counter.count_events(events),
                compacted_event_count=0,
                retained_event_count=len(events),
            )

        trajectory = events_to_trajectory(window.compacted)
        segmentation = self._run_stage(
            "segment",
            lambda: self._segment(trajectory),
            trajectory_chars=len(trajectory),
            compacted_event_count=len(window.compacted),
        )
        memory_document = sanitize_memory_document(
            self._run_stage(
                "synthesize",
                lambda: self._synthesize(trajectory, segmentation),
                trajectory_chars=len(trajectory),
                segmentation_chars=len(segmentation),
            )
        )

        critiques: list[Critique] = []
        revisions = 0
        local_report = quality_report(memory_document)
        if local_report is not None:
            critique = Critique(local_report)
            critiques.append(critique)
            self._emit_stage(
                "local_quality_violation",
                memory_chars=len(memory_document),
                critique_chars=len(critique.text),
            )
            memory_document = sanitize_memory_document(
                self._run_stage(
                    "revise",
                    lambda: self._revise(trajectory, memory_document, critique.text),
                    memory_chars=len(memory_document),
                    critique_chars=len(critique.text),
                    revision_index=revisions + 1,
                )
            )
            revisions += 1
        for loop_index in range(self.policy.max_critic_loops):
            critique = Critique(
                self._run_stage(
                    "critique",
                    lambda: self._critique(trajectory, memory_document),
                    memory_chars=len(memory_document),
                    critique_index=loop_index + 1,
                )
            )
            critiques.append(critique)
            if critique.approved:
                self._emit_stage(
                    "critique_approved",
                    critique_index=loop_index + 1,
                    critique_chars=len(critique.text),
                )
                break
            memory_document = sanitize_memory_document(
                self._run_stage(
                    "revise",
                    lambda: self._revise(trajectory, memory_document, critique.text),
                    memory_chars=len(memory_document),
                    critique_chars=len(critique.text),
                    revision_index=revisions + 1,
                )
            )
            revisions += 1

        memory_event = SessionEvent(
            type=EventType.USER,
            turn=window.compacted[0].turn,
            payload={
                "role": "user",
                "content": memory_restore_message(memory_document),
                "kind": "memory_restore",
                "compacted_event_count": len(window.compacted),
                "retained_event_count": len(window.retained),
            },
        )
        return CompactionResult(
            events=[memory_event, *window.retained],
            memory_document=memory_document,
            segmentation=segmentation,
            critiques=critiques,
            revisions=revisions,
            token_estimate_before=token_estimate or self.token_counter.count_events(events),
            compacted_event_count=len(window.compacted),
            retained_event_count=len(window.retained),
        )

    def _segment(self, trajectory: str) -> str:
        return self.generator.generate(SEGMENTATION_PROMPT, f"Trajectory:\n\n{trajectory}")

    def _synthesize(self, trajectory: str, segmentation: str) -> str:
        return self.generator.generate(
            SYNTHESIS_PROMPT,
            "\n\n".join(
                [
                    "Segmentation map:",
                    segmentation,
                    "Full trajectory:",
                    trajectory,
                ]
            ),
        )

    def _critique(self, trajectory: str, memory_document: str) -> str:
        return self.generator.generate(
            CRITIC_PROMPT,
            "\n\n".join(
                [
                    "Original trajectory:",
                    trajectory,
                    "Memory document:",
                    memory_document,
                ]
            ),
        )

    def _revise(self, trajectory: str, memory_document: str, critique: str) -> str:
        return self.generator.generate(
            REVISION_PROMPT,
            "\n\n".join(
                [
                    "Original trajectory:",
                    trajectory,
                    "Memory document:",
                    memory_document,
                    "Critic report:",
                    critique,
                ]
            ),
        )

    def _run_stage(
        self,
        name: str,
        fn: Callable[[], str],
        **payload: int | str,
    ) -> str:
        started = perf_counter()
        self._emit_stage(f"{name}_start", **payload)
        value = fn()
        elapsed_ms = int((perf_counter() - started) * 1000)
        self._emit_stage(
            f"{name}_end",
            elapsed_ms=elapsed_ms,
            output_chars=len(value),
            **payload,
        )
        return value

    def _emit_stage(self, name: str, **payload: int | str) -> None:
        if self.on_stage is None:
            return
        self.on_stage(name, payload)
