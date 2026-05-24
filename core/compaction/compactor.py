from __future__ import annotations

from core.compaction.llm import LangChainTextGenerator, TextGenerator
from core.compaction.models import CompactionResult, Critique
from core.compaction.policy import CompactionPolicy
from core.compaction.prompts import CRITIC_PROMPT, REVISION_PROMPT, SEGMENTATION_PROMPT, SYNTHESIS_PROMPT
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
    ) -> None:
        self.policy = policy or CompactionPolicy()
        self.policy.validate()
        self.generator = generator or LangChainTextGenerator.from_chat_model(self.policy.model)
        self.token_counter = token_counter or TokenCounter()

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
        segmentation = self._segment(trajectory)
        memory_document = self._synthesize(trajectory, segmentation)

        critiques: list[Critique] = []
        revisions = 0
        for _ in range(self.policy.max_critic_loops):
            critique = Critique(self._critique(trajectory, memory_document))
            critiques.append(critique)
            if critique.approved:
                break
            memory_document = self._revise(trajectory, memory_document, critique.text)
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
