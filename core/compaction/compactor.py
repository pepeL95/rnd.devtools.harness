from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from core.compaction.models import CompactionResult, Critique
from core.compaction.policy import CompactionPolicy
from core.compaction.prompts import CRITIC_PROMPT, REVISION_PROMPT, SEGMENTATION_PROMPT, SYNTHESIS_PROMPT
from core.compaction.quality import quality_report, sanitize_memory_document
from core.compaction.serialization import events_to_trajectory, memory_restore_message
from core.compaction.token_counter import TokenCounter
from core.compaction.window import split_compaction_window
from core.session.events import EventType, SessionEvent
from core.utilities.defaults import get_model_name


class Compactor:
    """Run the segmentation -> synthesis -> critic -> revision compaction loop."""

    def __init__(
        self,
        policy: CompactionPolicy | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.policy = policy or CompactionPolicy()
        self.policy.validate()
        self.token_counter = token_counter or TokenCounter()

    def compact(
        self,
        events: list[SessionEvent],
        token_estimate: int | None = None,
    ) -> CompactionResult:
        usage = _TokenUsage()
        window = split_compaction_window(events, self.policy.keep_last_turns)
        if not window.should_compact:
            return CompactionResult(
                events=list(events),
                memory_document="",
                segmentation="",
                model_names=_session_model_names(self.policy),
                token_usage=usage.as_dict(),
                token_estimate_before=token_estimate or self.token_counter.count_events(events),
                compacted_event_count=0,
                retained_event_count=len(events),
            )

        trajectory = events_to_trajectory(window.compacted)
        segmentation = _invoke_text(
            self.policy.task_extractor_model,
            SEGMENTATION_PROMPT,
            f"Trajectory:\n\n{trajectory}",
            token_counter=self.token_counter,
            usage=usage,
        )
        memory_document = sanitize_memory_document(
            _invoke_text(
                self.policy.compactor_model,
                SYNTHESIS_PROMPT,
                "\n\n".join(
                    [
                        "Segmentation map:",
                        segmentation,
                        "Full trajectory:",
                        trajectory,
                    ]
                ),
                token_counter=self.token_counter,
                usage=usage,
            )
        )

        critiques: list[Critique] = []
        revisions = 0
        local_report = quality_report(memory_document)
        if local_report is not None:
            critique = Critique(local_report)
            critiques.append(critique)
            memory_document = sanitize_memory_document(
                _invoke_text(
                    self.policy.compactor_model,
                    REVISION_PROMPT,
                    "\n\n".join(
                        [
                            "Original trajectory:",
                            trajectory,
                            "Memory document:",
                            memory_document,
                            "Critic report:",
                            critique.text,
                        ]
                    ),
                    token_counter=self.token_counter,
                    usage=usage,
                )
            )
            revisions += 1

        for _ in range(self.policy.max_critic_loops):
            critique = Critique(
                _invoke_text(
                    self.policy.critic_model,
                    CRITIC_PROMPT,
                    "\n\n".join(
                        [
                            "Original trajectory:",
                            trajectory,
                            "Memory document:",
                            memory_document,
                        ]
                    ),
                    token_counter=self.token_counter,
                    usage=usage,
                )
            )
            critiques.append(critique)
            if critique.approved:
                break
            memory_document = sanitize_memory_document(
                _invoke_text(
                    self.policy.compactor_model,
                    REVISION_PROMPT,
                    "\n\n".join(
                        [
                            "Original trajectory:",
                            trajectory,
                            "Memory document:",
                            memory_document,
                            "Critic report:",
                            critique.text,
                        ]
                    ),
                    token_counter=self.token_counter,
                    usage=usage,
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
            model_names=_session_model_names(self.policy),
            token_usage=usage.as_dict(),
            revisions=revisions,
            token_estimate_before=token_estimate or self.token_counter.count_events(events),
            compacted_event_count=len(window.compacted),
            retained_event_count=len(window.retained),
        )


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


def _session_model_names(policy: CompactionPolicy) -> dict[str, str]:
    return {
        "task_extractor": get_model_name(policy.task_extractor_model),
        "compactor": get_model_name(policy.compactor_model),
        "critic": get_model_name(policy.critic_model),
    }


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
