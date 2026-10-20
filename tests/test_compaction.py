import json
import sys
from threading import Event
from pathlib import Path
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from time import sleep
from unittest import TestCase
from unittest.mock import patch

from core.compaction.compactor import Compactor
from core.compaction.coordinator import CompactionCoordinator
from core.compaction.policy import CompactionPolicy
from core.compaction.serialization import memory_restore_message
from core.compaction.window import split_compaction_window
from core.middleware.compaction import CompactionMiddleware
from core.session.events import EventType, SessionEvent
from core.session.manager import SessionManager
from core.telemetry.store import TelemetryStore


def arc_update(turns=(1,), *, arc_id="", findings="Exact matching preserves tabs; normalization would corrupt indentation."):
    return {
        "narrative": "Preserve exact editing semantics across the session.",
        "current_state": "Continue checking indentation diagnostics without normalizing source.",
        "arcs": [{
            "arc_id": arc_id, "title": "Editing contract", "status": "completed",
            "intent": "Improve edit failures without changing exact-match semantics.",
            "implementation": "core/middleware/filesystem.py reports whitespace mismatches.",
            "findings": findings,
            "constraints": "Do not normalize source whitespace or add configuration.",
            "validation": "Focused regression tests passed.",
            "remaining": "Verify diagnostics on mixed tabs and spaces.",
            "turns": list(turns), "keywords": [],
        }],
    }


COMPLETE_ARC_UPDATE = json.dumps(arc_update())


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class ScriptedModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str]] = []

    def invoke(self, messages: list[object]) -> FakeResponse:
        self.calls.append((str(messages[0].content), str(messages[1].content)))
        if not self.responses:
            raise AssertionError("No scripted response left.")
        return FakeResponse(self.responses.pop(0))


class BlockingModel(ScriptedModel):
    def __init__(self, responses: list[str], started: Event, release: Event) -> None:
        super().__init__(responses)
        self.started = started
        self.release = release

    def invoke(self, messages: list[object]) -> FakeResponse:
        self.started.set()
        self.release.wait(timeout=2)
        return super().invoke(messages)


def scripted_session_policy(*, trigger_tokens: int = 1, keep_last_turns: int = 1, max_critic_loops: int = 2, trigger_on_day_change: bool = False) -> CompactionPolicy:
    model = ScriptedModel(
        [
            COMPLETE_ARC_UPDATE,
            "APPROVE",
        ]
    )
    return CompactionPolicy(
        trigger_tokens=trigger_tokens,
        keep_last_turns=keep_last_turns,
        max_critic_loops=max_critic_loops,
        trigger_on_day_change=trigger_on_day_change,
        compactor_model=model,
        critic_model=model,
    )


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
    def _wait_until_idle(self, coordinator: CompactionCoordinator, timeout_seconds: float = 2.0) -> None:
        deadline = datetime.now(timezone.utc) + timedelta(seconds=timeout_seconds)
        while coordinator.is_running() and datetime.now(timezone.utc) < deadline:
            sleep(0.01)

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
        model = ScriptedModel(
            [
                COMPLETE_ARC_UPDATE,
                "APPROVE",
            ]
        )
        compactor = Compactor(
            policy=CompactionPolicy(trigger_tokens=1, keep_last_turns=1, compactor_model=model, critic_model=model),
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
        self.assertIn("## Task Map", result.events[0].payload["content"])
        self.assertEqual(result.events[1].payload["content"], "continue")

    def test_compactor_runs_revision_loop_until_approved(self) -> None:
        model = ScriptedModel(
            [
                COMPLETE_ARC_UPDATE,
                "Preserve the concrete edit diagnostics.",
                json.dumps(arc_update(findings="Revised findings preserve exact edit diagnostics.")),
                "APPROVE",
            ]
        )
        compactor = Compactor(
            policy=CompactionPolicy(
                trigger_tokens=1,
                keep_last_turns=1,
                max_critic_loops=2,
                compactor_model=model,
                critic_model=model,
            ),
        )

        result = compactor.compact(
            [
                event(1, EventType.USER, "original task"),
                event(2, EventType.USER, "latest"),
            ]
        )

        self.assertEqual(result.revisions, 1)
        self.assertIn("Revised findings preserve exact edit diagnostics.", result.memory_document)
        self.assertEqual(len(result.critiques), 2)

    def test_compactor_repairs_invalid_arc_before_model_critic(self) -> None:
        model = ScriptedModel(["not JSON", COMPLETE_ARC_UPDATE, "APPROVE"])
        compactor = Compactor(policy=CompactionPolicy(
            keep_last_turns=1, compactor_model=model, critic_model=model,
        ))
        result = compactor.compact([event(1, EventType.USER, "original task"), event(2, EventType.USER, "latest")])
        self.assertEqual(result.revisions, 1)
        self.assertEqual(len(result.critiques), 1)
        self.assertIn("Previous output failed validation", model.calls[1][1])
        self.assertIn("## Task Map", result.memory_document)

    def test_compactor_rejects_process_artifacts_outside_arc_schema(self) -> None:
        draft = arc_update()
        draft["revision_log"] = "Removed noise"
        model = ScriptedModel([json.dumps(draft), COMPLETE_ARC_UPDATE, "APPROVE"])
        compactor = Compactor(policy=CompactionPolicy(
            keep_last_turns=1, compactor_model=model, critic_model=model,
        ))
        result = compactor.compact([event(1, EventType.USER, "original task"), event(2, EventType.USER, "latest")])
        self.assertEqual(result.revisions, 1)
        self.assertNotIn("revision_log", result.memory_document)
        self.assertTrue(result.memory_document.startswith("# Session Arc"))

    def test_compactor_uses_only_synthesis_and_review_models(self) -> None:
        compactor_model = ScriptedModel([COMPLETE_ARC_UPDATE])
        critic_model = ScriptedModel(["APPROVE"])
        compactor = Compactor(policy=CompactionPolicy(
            keep_last_turns=1, compactor_model=compactor_model, critic_model=critic_model,
        ))
        result = compactor.compact([event(1, EventType.USER, "original task"), event(2, EventType.USER, "latest")])
        self.assertEqual(result.revisions, 0)
        self.assertEqual(len(compactor_model.calls), 1)
        self.assertEqual(len(critic_model.calls), 1)
        self.assertEqual(result.token_usage["call_count"], 2)
        self.assertEqual(set(result.model_names), {"compactor", "critic"})

    def test_repeated_compaction_preserves_entire_task_set_without_bank(self) -> None:
        initial = arc_update()
        second_task = arc_update((2,), findings="Cancellation resumes the same turn.")["arcs"][0]
        second_task.update(title="Cancellation", status="active", remaining="Check live steering")
        initial["arcs"].append(second_task)
        model = ScriptedModel([json.dumps(initial)])
        compactor = Compactor(policy=CompactionPolicy(
            keep_last_turns=1, compactor_model=model, max_critic_loops=0,
        ))
        events = [event(turn, EventType.USER, f"Task {turn}") for turn in range(1, 4)]
        with patch.dict(sys.modules, {"core.trajectory_bank.archival": None, "chromadb": None}):
            first = compactor.compact(events)
            original_arcs = first.events[0].payload["task_arcs"]

            # Work on a third task must not erase either earlier task.
            third_task = arc_update((3,), findings="Configuration is workspace-local.")
            third_task["arcs"][0]["title"] = "Configuration"
            model.responses.append(json.dumps(third_task))
            second = compactor.compact([*first.events, event(4, EventType.USER, "Resume cancellation")])
            self.assertEqual(second.events[0].payload["task_arcs"][:2], original_arcs)

            # A nonconsecutive return updates the same arc, retaining its historical turn membership.
            updated = dict(original_arcs[1], turns=[4], status="completed",
                           findings="Cancellation resumes the same turn; live steering verified.", remaining="None known")
            model.responses.append(json.dumps({**third_task, "arcs": [updated]}))
            third = compactor.compact([*second.events, event(5, EventType.USER, "Next task")])

        arcs = third.events[0].payload["task_arcs"]
        self.assertEqual(len(arcs), 3)
        self.assertEqual(arcs[0], original_arcs[0])
        self.assertEqual(arcs[2], second.events[0].payload["task_arcs"][2])
        self.assertEqual(arcs[1]["arc_id"], original_arcs[1]["arc_id"])
        self.assertEqual(arcs[1]["turns"], [2, 4])
        self.assertEqual(arcs[1]["status"], "completed")
        self.assertIn("normalization would corrupt indentation", third.memory_document)
        self.assertIn("live steering verified", third.memory_document)
        self.assertEqual(third.events[-1].turn, 5)
        self.assertEqual(third.events[0].payload["trajectory_manifest"], {})
        self.assertNotIn("trajectory://", third.memory_document)
        prompt = json.loads(model.calls[1][1])
        self.assertEqual(prompt["existing_arcs"], original_arcs)
        self.assertEqual(prompt["previous_session_context"], [{
            "narrative": initial["narrative"], "current_state": initial["current_state"],
        }])
        self.assertEqual(prompt["new_turns"], [3])
        self.assertIn("Resume cancellation", prompt["retained_tail_for_orientation_only"])

    def test_legacy_memory_survives_bankless_compaction(self) -> None:
        old = event(0, EventType.USER, "Historical requirement: never rewrite user changes.")
        old.payload["kind"] = "memory_restore"
        compactor = Compactor(policy=scripted_session_policy())
        result = compactor.compact([old, event(1, EventType.USER, "New work"), event(2, EventType.USER, "Tail")])
        self.assertIn(old.payload["content"], result.memory_document)
        self.assertTrue(result.events[0].payload["task_arcs"][0]["arc_id"].startswith("legacy-"))

    def test_no_eviction_does_not_call_models(self) -> None:
        model = ScriptedModel([])
        compactor = Compactor(policy=CompactionPolicy(keep_last_turns=2, compactor_model=model))
        events = [event(1, EventType.USER, "Keep this")]
        for source in ([], events):
            self.assertEqual(compactor.compact(source).events, source)
        self.assertEqual(model.calls, [])

    def test_semantic_eval_scores_arc_sections_and_concrete_findings(self) -> None:
        from evals.compaction_semantic_richness import _score_case

        result = Compactor(policy=scripted_session_policy()).compact([
            event(1, EventType.USER, "Keep exact whitespace matching"), event(2, EventType.USER, "Continue"),
        ])
        case = {"id": "arc-format", "title": "Arc format", "expected_atoms": [
            {"id": "whitespace", "label": "Preserve exact matching", "must_include_any": ["normalization would corrupt indentation"]},
            {"id": "missing", "label": "Missing evidence stays missing", "must_include_any": ["MISSING_EVIDENCE_SENTINEL"]},
        ]}
        score = _score_case(case, result.memory_document, result.memory_document, result)
        self.assertEqual(score.section_score, 1.0)
        self.assertEqual(score.atom_score, 0.5)
        self.assertEqual(score.missing_sections, [])

    def test_compaction_middleware_runs_compaction_in_background(self) -> None:
        with TemporaryDirectory() as directory:
            manager = SessionManager(session_id="s1", root=Path(directory))
            manager.append(
                [
                    event(1, EventType.USER, "old"),
                    event(2, EventType.USER, "new"),
                ]
            )
            compactor = Compactor(
                policy=scripted_session_policy(),
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
            self._wait_until_idle(coordinator)

            curated = manager.read_curated()
            self.assertEqual(curated[0].payload["kind"], "memory_restore")
            self.assertIn("[MEMORY RESTORE]", curated[0].payload["content"])
            self.assertEqual(curated[1].payload["content"], "new")
            self.assertTrue(telemetry_path.exists())
            entries = self._compaction_telemetry_entries(telemetry_path)
            self.assertEqual(entries[0]["name"], "compaction.start")
            self.assertEqual(entries[-1]["name"], "compaction.end")
            self.assertEqual(entries[-1]["payload"]["counts"]["compacted_event_count"], 1)
            self.assertEqual(entries[-1]["payload"]["kind"], "session")
            self.assertEqual(entries[-1]["payload"]["token_usage"]["source_tokens"], 999)

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
            model = ScriptedModel(
                [
                    COMPLETE_ARC_UPDATE,
                    "APPROVE",
                ]
            )
            compactor = Compactor(
                policy=CompactionPolicy(trigger_tokens=1, keep_last_turns=1, compactor_model=model, critic_model=model),
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
            self.assertEqual(status, "started")
            self._wait_until_idle(coordinator)

            self.assertEqual([phase for phase, _ in observed], ["start", "end"])
            self.assertEqual(observed[0][1]["token_usage"]["source_tokens"], 999)
            self.assertEqual(observed[1][1]["counts"]["compacted_event_count"], 1)
            self.assertEqual(observed[1][1]["model_names"]["compactor"], "ScriptedModel")
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
                policy=scripted_session_policy(),
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

            self.assertEqual(status, "started")
            self._wait_until_idle(coordinator)
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
                policy=scripted_session_policy(),
            )
            started = Event()
            release = Event()
            blocking_model = BlockingModel(
                [
                    COMPLETE_ARC_UPDATE,
                    "APPROVE",
                ],
                started=started,
                release=release,
            )
            coordinator = CompactionCoordinator(
                manager,
                Compactor(
                    policy=CompactionPolicy(
                        trigger_tokens=1,
                        keep_last_turns=1,
                        compactor_model=blocking_model,
                        critic_model=blocking_model,
                    )
                ),
                token_counter=FixedTokenCounter(999),
                telemetry_store=TelemetryStore(Path(directory) / "telemetry.jsonl"),
                repo_root=Path(directory),
            )

            first = coordinator.request_manual_compaction()
            self.assertEqual(first, "started")
            self.assertTrue(started.wait(timeout=1))

            status = coordinator.request_manual_compaction()

            self.assertEqual(status, "running")
            release.set()
            self._wait_until_idle(coordinator)

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
                policy=scripted_session_policy(trigger_tokens=1000, trigger_on_day_change=True),
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
            self.assertEqual(status, "started")
            self._wait_until_idle(coordinator)

            self.assertEqual(manager.read_curated()[0].payload["kind"], "memory_restore")
            self.assertEqual(
                [entry["name"] for entry in self._compaction_telemetry_entries(telemetry_path)],
                ["compaction.start", "compaction.end"],
            )

    def test_compaction_coordinator_preserves_newer_turns_added_while_running(self) -> None:
        with TemporaryDirectory() as directory:
            manager = SessionManager(session_id="s1", root=Path(directory))
            manager.append(
                [
                    event(1, EventType.USER, "old"),
                    event(2, EventType.USER, "new"),
                ]
            )
            started = Event()
            release = Event()
            blocking_model = BlockingModel(
                [
                    COMPLETE_ARC_UPDATE,
                    "APPROVE",
                ],
                started=started,
                release=release,
            )
            coordinator = CompactionCoordinator(
                manager,
                Compactor(
                    policy=CompactionPolicy(
                        trigger_tokens=1,
                        keep_last_turns=1,
                        compactor_model=blocking_model,
                        critic_model=blocking_model,
                    )
                ),
                token_counter=FixedTokenCounter(999),
                telemetry_store=TelemetryStore(Path(directory) / "telemetry.jsonl"),
                repo_root=Path(directory),
            )

            status = coordinator.request_manual_compaction()
            self.assertEqual(status, "started")
            self.assertTrue(started.wait(timeout=1))
            manager.append([event(3, EventType.USER, "latest")])
            release.set()
            self._wait_until_idle(coordinator)

            curated = manager.read_curated()
            self.assertEqual(curated[0].payload["kind"], "memory_restore")
            self.assertIn("Exact matching preserves tabs", curated[0].payload["content"])
            self.assertEqual([item.payload["content"] for item in curated[1:]], ["new", "latest"])

    def test_memory_restore_message_matches_injection_contract(self) -> None:
        content = memory_restore_message("DOC")

        self.assertTrue(content.startswith("[MEMORY RESTORE]"))
        self.assertIn("DOC", content)
        self.assertIn("[END MEMORY RESTORE]", content)
        self.assertTrue(content.endswith("Your task continues below."))
