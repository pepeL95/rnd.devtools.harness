from pathlib import Path
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest import TestCase

from core.compaction.compactor import Compactor
from core.compaction.policy import CompactionPolicy
from core.compaction.serialization import memory_restore_message
from core.compaction.window import split_compaction_window
from core.middleware.compaction import CompactionMiddleware
from core.session.events import EventType, SessionEvent
from core.session.session_manager import SessionManager


COMPLETE_MEMORY_DOCUMENT = """## Episodic Memory

### Task Requirement Synthesis
#### Compaction contract
- FULL-FIDELITY REF: turns [1-2]
- TASK TIMESTAMPS: 2026-05-23T10:00:00+00:00 -> 2026-05-23T10:15:00+00:00
- TASK REQUEST SYNTHESIS: Stabilize compaction output so older trajectory state can be compressed without losing the exact continuation edge for the next turn.
- TASK EXECUTION SYNTHESIS: The session converged on a compaction design where resumability matters more than transcript fidelity. The important state transition was moving from raw history retention to a synthesized memory record that still preserves constraints, current state, and the lens the next agent needs when it enters the preserved tail.
- PRIORITY SIGNALS: preserve exact file paths, favor semantic synthesis, keep the resumption contract intact.
- OPEN LOOP: Validate that revisions keep improving semantic density without eroding the restore contract.

### Current State
- `core/compaction/prompts.py` is modified and tests are passing.
- The compactor is expected to replace older curated events with a memory restore record while retaining the latest turn.

### Work Completed
- Implemented the compaction pipeline and verified the memory restore injection contract in tests.

### Failed Approaches
- APPROACH: Using transcript-like summaries with weak structure.
- FAILURE MECHANISM: The result preserves wording but loses reusable judgment and leaves the next agent to reread the trajectory.
- REUSABLE LESSON: Compaction should preserve mechanisms, constraints, and state transitions instead of replaying the session.
- STATUS: abandoned

### Open Problems
- No active blocker in this fixture. The remaining question is whether the critic loop keeps the summary dense enough.

### Implicit Tasks Discovered
- The compactor needs a stable heading contract so downstream tooling can reason about the memory document.

### Next Steps
- Keep validating that compaction revisions preserve resumability while increasing semantic density.

## Semantic Memory

### Codebase Characteristics
- The session layer stores curated history as JSONL events and expects compaction to reinsert a synthetic user memory record.

### Task-Approach Pairs
- TASK CLASS: trajectory compaction for coding agents
- EFFECTIVE APPROACH: use segmentation plus a critique loop that pushes toward mechanism-rich summaries.
- PITFALLS: transcript replay, vague failed approaches, and missing exact file paths.
- CONFIDENCE: high

### Generalizable Insights
- Compaction quality improves when the system optimizes for causal structure and next-step utility instead of preserving long verbatim spans. This transfers to any agent that resumes from compressed state. Confidence: high.

## Handoff

### Session Narrative
The fixture represents a compaction run where the important requirement is structural fidelity plus semantic density. The exact wording is not important; the resumable state and learned constraints are.

The next agent should be able to continue without rereading the raw trajectory because the durable constraints, current state, and continuation plan are all explicit.
"""


class ScriptedGenerator:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if not self.responses:
            raise AssertionError("No scripted response left.")
        return self.responses.pop(0)


class FixedTokenCounter:
    def __init__(self, count: int) -> None:
        self.count = count

    def count_events(self, events: object) -> int:
        return self.count


def event(turn: int, event_type: EventType, content: str) -> SessionEvent:
    role = "assistant" if event_type == EventType.ASSISTANT else "user"
    return SessionEvent(type=event_type, turn=turn, payload={"role": role, "content": content})


def dated_event(turn: int, when: datetime, content: str = "x") -> SessionEvent:
    return SessionEvent(
        type=EventType.USER,
        turn=turn,
        timestamp=when.isoformat(),
        payload={"role": "user", "content": content},
    )


class CompactionTests(TestCase):
    def test_compaction_policy_triggers_on_age(self) -> None:
        policy = CompactionPolicy(trigger_tokens=1000, trigger_after=timedelta(hours=2))
        events = [dated_event(1, datetime(2026, 5, 23, 10, tzinfo=timezone.utc))]

        decision = policy.compaction_decision(
            10,
            events=events,
            now=datetime(2026, 5, 23, 13, tzinfo=timezone.utc),
        )

        self.assertTrue(decision.should_compact)
        self.assertEqual(decision.reason, "age")

    def test_compaction_policy_triggers_on_day_change(self) -> None:
        policy = CompactionPolicy(trigger_tokens=1000, trigger_on_day_change=True)
        events = [dated_event(1, datetime(2026, 5, 22, 23, tzinfo=timezone.utc))]

        decision = policy.compaction_decision(
            10,
            events=events,
            now=datetime(2026, 5, 23, 1, tzinfo=timezone.utc),
        )

        self.assertTrue(decision.should_compact)
        self.assertEqual(decision.reason, "day_change")

    def test_split_compaction_window_keeps_last_k_turns(self) -> None:
        events = [
            event(1, EventType.USER, "u1"),
            event(1, EventType.ASSISTANT, "a1"),
            event(2, EventType.USER, "u2"),
            event(3, EventType.USER, "u3"),
        ]

        window = split_compaction_window(events, keep_last_turns=2)

        self.assertEqual([item.turn for item in window.compacted], [1, 1])
        self.assertEqual([item.turn for item in window.retained], [2, 3])

    def test_compactor_replaces_old_events_with_memory_restore(self) -> None:
        generator = ScriptedGenerator(
            [
                "EPISODE 1: setup\nTURNS: 1 to 1",
                COMPLETE_MEMORY_DOCUMENT,
                "CRITIQUE SUMMARY\n  VIOLATIONS FOUND: 0\n  RECOMMENDED ACTION: approve as-is",
            ]
        )
        compactor = Compactor(
            generator=generator,
            policy=CompactionPolicy(trigger_tokens=1, keep_last_turns=1),
        )
        events = [
            event(1, EventType.USER, "original task"),
            event(1, EventType.ASSISTANT, "done"),
            event(2, EventType.USER, "continue"),
        ]

        result = compactor.compact(events, token_estimate=100)

        self.assertEqual(result.compacted_event_count, 2)
        self.assertEqual(result.retained_event_count, 1)
        self.assertEqual(len(result.events), 2)
        self.assertEqual(result.events[0].payload["kind"], "memory_restore")
        self.assertIn("[MEMORY RESTORE]", result.events[0].payload["content"])
        self.assertIn("### Task Requirement Synthesis", result.events[0].payload["content"])
        self.assertEqual(result.events[1].payload["content"], "continue")

    def test_compactor_runs_revision_loop_until_approved(self) -> None:
        generator = ScriptedGenerator(
            [
                "EPISODE 1: setup\nTURNS: 1 to 1",
                COMPLETE_MEMORY_DOCUMENT,
                "CRITIQUE SUMMARY\n  VIOLATIONS FOUND: 1\n  RECOMMENDED ACTION: revise targeted sections",
                COMPLETE_MEMORY_DOCUMENT.replace("tests are passing", "tests remain passing after revision"),
                "CRITIQUE SUMMARY\n  VIOLATIONS FOUND: 0\n  RECOMMENDED ACTION: approve as-is",
            ]
        )
        compactor = Compactor(
            generator=generator,
            policy=CompactionPolicy(trigger_tokens=1, keep_last_turns=1, max_critic_loops=2),
        )

        result = compactor.compact(
            [
                event(1, EventType.USER, "original task"),
                event(2, EventType.USER, "latest"),
            ]
        )

        self.assertEqual(result.revisions, 1)
        self.assertIn("tests remain passing after revision", result.memory_document)
        self.assertEqual(len(result.critiques), 2)

    def test_compactor_revises_legacy_quote_heavy_draft_before_model_critic(self) -> None:
        generator = ScriptedGenerator(
            [
                "EPISODE 1: setup\nTURNS: 1 to 1",
                'ORIGINAL TASK\n"repeat the whole prompt verbatim because that feels safe"\nUSER DIRECTIVES\n"quote everything back"',
                COMPLETE_MEMORY_DOCUMENT,
                "CRITIQUE SUMMARY\n  VIOLATIONS FOUND: 0\n  RECOMMENDED ACTION: approve as-is",
            ]
        )
        compactor = Compactor(
            generator=generator,
            policy=CompactionPolicy(trigger_tokens=1, keep_last_turns=1),
        )

        result = compactor.compact(
            [
                event(1, EventType.USER, "original task"),
                event(2, EventType.USER, "latest"),
            ]
        )

        self.assertEqual(result.revisions, 1)
        self.assertEqual(len(result.critiques), 2)
        self.assertIn("LOCAL QUALITY REVIEW", result.critiques[0].text)
        self.assertIn("### Task Requirement Synthesis", result.memory_document)

    def test_compaction_middleware_replaces_curated_history(self) -> None:
        with TemporaryDirectory() as directory:
            manager = SessionManager(session_id="s1", root=Path(directory))
            manager.append(
                [
                    event(1, EventType.USER, "old"),
                    event(2, EventType.USER, "new"),
                ]
            )
            compactor = Compactor(
                generator=ScriptedGenerator(
                    [
                        "EPISODE 1: old\nTURNS: 1 to 1",
                        COMPLETE_MEMORY_DOCUMENT,
                        "CRITIQUE SUMMARY\n  VIOLATIONS FOUND: 0\n  RECOMMENDED ACTION: approve as-is",
                    ]
                ),
                policy=CompactionPolicy(trigger_tokens=1, keep_last_turns=1),
            )
            middleware = CompactionMiddleware(manager, compactor, token_counter=FixedTokenCounter(999))

            middleware.after_agent({"messages": []}, runtime=None)

            curated = manager.read_curated()
            self.assertEqual(len(curated), 2)
            self.assertEqual(curated[0].payload["kind"], "memory_restore")
            self.assertEqual(curated[1].payload["content"], "new")

    def test_compaction_middleware_compacts_before_session_load(self) -> None:
        with TemporaryDirectory() as directory:
            manager = SessionManager(session_id="s1", root=Path(directory))
            manager.append(
                [
                    event(1, EventType.USER, "old"),
                    event(2, EventType.USER, "new"),
                ]
            )
            compactor = Compactor(
                generator=ScriptedGenerator(
                    [
                        "EPISODE 1: old\nTURNS: 1 to 1",
                        COMPLETE_MEMORY_DOCUMENT,
                        "CRITIQUE SUMMARY\n  VIOLATIONS FOUND: 0\n  RECOMMENDED ACTION: approve as-is",
                    ]
                ),
                policy=CompactionPolicy(trigger_tokens=1, keep_last_turns=1),
            )
            middleware = CompactionMiddleware(manager, compactor, token_counter=FixedTokenCounter(999))

            middleware.before_agent({"messages": []}, runtime=None)

            curated = manager.read_curated()
            self.assertEqual(curated[0].payload["kind"], "memory_restore")

    def test_compaction_middleware_uses_date_policy(self) -> None:
        with TemporaryDirectory() as directory:
            manager = SessionManager(session_id="s1", root=Path(directory))
            manager.append(
                [
                    dated_event(1, datetime(2026, 5, 22, 23, tzinfo=timezone.utc), "old"),
                    dated_event(2, datetime(2026, 5, 22, 23, 30, tzinfo=timezone.utc), "new"),
                ]
            )
            compactor = Compactor(
                generator=ScriptedGenerator(
                    [
                        "EPISODE 1: old\nTURNS: 1 to 1",
                        COMPLETE_MEMORY_DOCUMENT,
                        "CRITIQUE SUMMARY\n  VIOLATIONS FOUND: 0\n  RECOMMENDED ACTION: approve as-is",
                    ]
                ),
                policy=CompactionPolicy(trigger_tokens=1000, keep_last_turns=1, trigger_on_day_change=True),
            )
            middleware = CompactionMiddleware(manager, compactor, token_counter=FixedTokenCounter(1))

            runtime = SimpleNamespace(context={"now": datetime(2026, 5, 23, 1, tzinfo=timezone.utc)})
            middleware.before_agent({"messages": []}, runtime=runtime)

            self.assertEqual(manager.read_curated()[0].payload["kind"], "memory_restore")

    def test_memory_restore_message_matches_injection_contract(self) -> None:
        content = memory_restore_message("DOC")

        self.assertTrue(content.startswith("[MEMORY RESTORE]"))
        self.assertIn("DOC", content)
        self.assertIn("[END MEMORY RESTORE]", content)
        self.assertTrue(content.endswith("Your task continues below."))
