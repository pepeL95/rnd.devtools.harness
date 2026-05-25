import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest import TestCase

from core.compaction.compactor import Compactor
from core.compaction.coordinator import CompactionCoordinator
from core.compaction.policy import CompactionPolicy
from core.compaction.serialization import memory_restore_message
from core.compaction.window import split_compaction_window
from core.middleware.compaction import CompactionMiddleware
from core.session.events import EventType, SessionEvent
from core.session.manager import SessionManager
from core.telemetry.store import TelemetryStore


COMPLETE_MEMORY_DOCUMENT = """## Episodic Memory

### Task History
#### Compaction contract
- FULL-FIDELITY REF: turns [1-2]
- TASK TIMESTAMPS: 2026-05-23T10:00:00+00:00 -> 2026-05-23T10:15:00+00:00
- TASK DESCRIPTION SYNTHESIS: Stabilize compaction output so older trajectory state can be compressed without losing the exact continuation edge for the next turn, while keeping the memory resumable and semantically rich.
- EXECUTION MEMORY: The task evolved from a raw-history framing into a resumability-first compaction design. The important thing to remember is not the chronology of edits but the result: older turns can collapse into a compact memory that still carries the live continuation edge, the meaningful constraints, and the context a later agent needs to re-enter the work without rereading the full transcript.

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
    def _compaction_telemetry_entries(self, path: Path) -> list[dict[str, object]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

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
        self.assertIn("### Task History", result.events[0].payload["content"])
        self.assertEqual(result.events[1].payload["content"], "continue")

    def test_compactor_runs_revision_loop_until_approved(self) -> None:
        generator = ScriptedGenerator(
            [
                "EPISODE 1: setup\nTURNS: 1 to 1",
                COMPLETE_MEMORY_DOCUMENT,
                "CRITIQUE SUMMARY\n  VIOLATIONS FOUND: 1\n  RECOMMENDED ACTION: revise targeted sections",
                COMPLETE_MEMORY_DOCUMENT.replace(
                    "preserve resumability while increasing semantic density.",
                    "preserve resumability while increasing semantic density after revision.",
                ),
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
        self.assertIn("preserve resumability while increasing semantic density after revision.", result.memory_document)
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
        self.assertIn("### Task History", result.memory_document)

    def test_compactor_strips_revision_artifacts_from_final_memory(self) -> None:
        generator = ScriptedGenerator(
            [
                "EPISODE 1: setup\nTURNS: 1 to 1",
                "junk before heading\n\n## Episodic Memory\n\n### Task History\n#### T\n- FULL-FIDELITY REF: turns [1]\n- TASK TIMESTAMPS: 2026-05-23T10:00:00+00:00 -> 2026-05-23T10:01:00+00:00\n- TASK DESCRIPTION SYNTHESIS: summarize the requested task with enough concrete detail for resumption and keep the real user intent visible in a task-aware way\n- EXECUTION MEMORY: this task changed the approach and captured the important session semantics in prose, preserving the meaningful findings and results without replaying low-signal operational trivia or critic-side bookkeeping that would pollute the final memory surface\n\n### Failed Approaches\n- none\n\n### Open Problems\n- none\n\n### Implicit Tasks Discovered\n- none\n\n### Next Steps\n- continue\n\n## Semantic Memory\n\n### Task-Approach Pairs\n- TASK CLASS: compact session memory\n- EFFECTIVE APPROACH: preserve signal\n- PITFALLS: replaying noise\n- CONFIDENCE: medium\n\n### Generalizable Insights\n- High-signal memory is more useful than transcript replay when the next agent must resume quickly.\n\n## Handoff\n\n### Session Narrative\nhandoff\n\nREVISION LOG\n[VIOLATION 1] -> removed noise",
                "CRITIQUE SUMMARY\n  VIOLATIONS FOUND: 0\n  RECOMMENDED ACTION: approve as-is",
            ]
        )
        compactor = Compactor(generator=generator, policy=CompactionPolicy(trigger_tokens=1, keep_last_turns=1))

        result = compactor.compact([event(1, EventType.USER, "original task"), event(2, EventType.USER, "latest")])

        self.assertNotIn("REVISION LOG", result.memory_document)
        self.assertNotIn("junk before heading", result.memory_document)
        self.assertTrue(result.memory_document.startswith("## Episodic Memory"))

    def test_compaction_middleware_runs_compaction_synchronously(self) -> None:
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
            telemetry_path = Path(directory) / "telemetry.jsonl"
            coordinator = CompactionCoordinator(
                manager,
                compactor,
                token_counter=FixedTokenCounter(999),
                telemetry_store=TelemetryStore(telemetry_path),
                repo_root=Path(directory),
            )
            middleware = CompactionMiddleware(coordinator)

            middleware.after_agent({"messages": []}, runtime=None)

            curated = manager.read_curated()
            self.assertEqual(curated[0].payload["kind"], "memory_restore")
            self.assertIn("[MEMORY RESTORE]", curated[0].payload["content"])
            self.assertEqual(curated[1].payload["content"], "new")
            self.assertTrue(telemetry_path.exists())
            entries = self._compaction_telemetry_entries(telemetry_path)
            self.assertEqual(entries[0]["name"], "compaction.start")
            self.assertEqual(entries[-1]["name"], "compaction.end")
            self.assertEqual(entries[-1]["payload"]["compacted_event_count"], 1)

    def test_compaction_coordinator_emits_lifecycle_events(self) -> None:
        with TemporaryDirectory() as directory:
            manager = SessionManager(session_id="s1", root=Path(directory))
            manager.append(
                [
                    event(1, EventType.USER, "old"),
                    event(2, EventType.USER, "new"),
                ]
            )
            observed: list[tuple[str, dict[str, object]]] = []
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
            telemetry_path = Path(directory) / "telemetry.jsonl"
            coordinator = CompactionCoordinator(
                manager,
                compactor,
                token_counter=FixedTokenCounter(999),
                on_compaction_event=lambda phase, payload: observed.append((phase, payload)),
                telemetry_store=TelemetryStore(telemetry_path),
            )

            status = coordinator.request_policy_compaction(runtime=None)
            self.assertEqual(status, "completed")

            self.assertEqual([phase for phase, _ in observed], ["start", "end"])
            self.assertEqual(observed[0][1]["estimated_tokens"], 999)
            self.assertEqual(observed[1][1]["compacted_event_count"], 1)
            self.assertEqual(manager.read_curated()[0].payload["kind"], "memory_restore")
            self.assertEqual(manager.read_curated()[1].payload["content"], "new")
            self.assertEqual(
                [entry["name"] for entry in self._compaction_telemetry_entries(telemetry_path)],
                ["compaction.start", "compaction.end"],
            )

    def test_compaction_coordinator_ignores_ui_callback_errors_after_successful_rewrite(self) -> None:
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
            telemetry_path = Path(directory) / "telemetry.jsonl"
            def failing_callback(phase: str, payload: dict[str, object]) -> None:
                if phase == "end":
                    raise RuntimeError("App is not running")

            coordinator = CompactionCoordinator(
                manager,
                compactor,
                token_counter=FixedTokenCounter(999),
                on_compaction_event=failing_callback,
                telemetry_store=TelemetryStore(telemetry_path),
                repo_root=Path(directory),
            )

            status = coordinator.request_manual_compaction()

            self.assertEqual(status, "completed")
            curated = manager.read_curated()
            self.assertEqual(curated[0].payload["kind"], "memory_restore")
            self.assertEqual(curated[1].payload["content"], "new")

            entries = self._compaction_telemetry_entries(telemetry_path)
            self.assertEqual([entry["name"] for entry in entries], ["compaction.start", "compaction.end"])

    def test_compaction_coordinator_reports_running_when_busy(self) -> None:
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
            coordinator = CompactionCoordinator(manager, compactor, token_counter=FixedTokenCounter(999))
            assert coordinator._run_lock.acquire(blocking=False)

            status = coordinator.request_manual_compaction()

            self.assertEqual(status, "running")
            coordinator._run_lock.release()

    def test_compaction_coordinator_uses_date_policy(self) -> None:
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
            telemetry_path = Path(directory) / "telemetry.jsonl"
            coordinator = CompactionCoordinator(
                manager,
                compactor,
                token_counter=FixedTokenCounter(1),
                telemetry_store=TelemetryStore(telemetry_path),
                repo_root=Path(directory),
            )

            runtime = SimpleNamespace(context={"now": datetime(2026, 5, 23, 1, tzinfo=timezone.utc)})
            status = coordinator.request_policy_compaction(runtime=runtime)
            self.assertEqual(status, "completed")

            self.assertEqual(manager.read_curated()[0].payload["kind"], "memory_restore")
            self.assertEqual(
                [entry["name"] for entry in self._compaction_telemetry_entries(telemetry_path)],
                ["compaction.start", "compaction.end"],
            )

    def test_memory_restore_message_matches_injection_contract(self) -> None:
        content = memory_restore_message("DOC")

        self.assertTrue(content.startswith("[MEMORY RESTORE]"))
        self.assertIn("DOC", content)
        self.assertIn("[END MEMORY RESTORE]", content)
        self.assertTrue(content.endswith("Your task continues below."))
