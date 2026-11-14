from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from time import sleep
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from core.compaction.token_counter import TokenCounter
from core.session.events import EventType, SessionEvent
from core.session.manager import SessionManager
from core.telemetry.store import TelemetryStore
from core.trajectory.compactor import TrajectoryCompactor
from core.trajectory.coordinator import TrajectoryCompactionCoordinator
from core.trajectory.models import TurnTrajectorySynthesis
from core.trajectory.policy import TrajectoryCompactionPolicy
from core.trajectory.serialization import format_turn_interval, trajectory_memory_message
from core.utilities.defaults import get_model_name


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class ScriptedModel:
    def __init__(self, response: str | list[str]) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def invoke(self, messages: list[object]) -> FakeResponse:
        self.calls.append((str(messages[0].content), str(messages[1].content)))
        response = self.response[len(self.calls) - 1] if isinstance(self.response, list) else self.response
        return FakeResponse(response)


class BlockingModel(ScriptedModel):
    def __init__(self, response: str | list[str], started: Event, release: Event) -> None:
        super().__init__(response)
        self.started = started
        self.release = release

    def invoke(self, messages: list[object]) -> FakeResponse:
        self.started.set()
        self.release.wait(timeout=2)
        return super().invoke(messages)


def scripted_trajectory_policy(response: str | list[str], *, trigger_every_turns: int = 2) -> tuple[TrajectoryCompactionPolicy, ScriptedModel]:
    model = ScriptedModel(response)
    return TrajectoryCompactionPolicy(trigger_every_turns=trigger_every_turns, compactor_model=model), model


def event(turn: int, event_type: EventType, content: str, **payload: object) -> SessionEvent:
    role = payload.pop("role", "assistant" if event_type == EventType.ASSISTANT else "user")
    return SessionEvent(type=event_type, turn=turn, payload={"role": role, "content": content, **payload})


class TrajectoryCompactionTests(TestCase):
    def setUp(self) -> None:
        # Isolate lifecycle tests from the size of their tiny fixture strings.
        # Real token accounting and the no-growth guard are exercised below.
        class FlowTokenCounter(TokenCounter):
            def count_context(self, events):
                return sum(1 if item.payload.get("kind") == "trajectory_memory" else 100 for item in events)

        counter_patch = patch("core.trajectory.compactor.TokenCounter", FlowTokenCounter)
        counter_patch.start()
        self.addCleanup(counter_patch.stop)

    def test_policy_triggers_every_n_turns(self) -> None:
        policy = TrajectoryCompactionPolicy(trigger_every_turns=2)

        self.assertFalse(policy.compaction_decision(turn_count=1, latest_turn=1).should_compact)
        self.assertTrue(policy.compaction_decision(turn_count=2, latest_turn=2).should_compact)
        self.assertFalse(policy.compaction_decision(turn_count=3, latest_turn=3).should_compact)
        self.assertTrue(policy.compaction_decision(turn_count=4, latest_turn=4).should_compact)

    def test_policy_exposes_dedicated_compactor_model_slot(self) -> None:
        policy = TrajectoryCompactionPolicy()

        self.assertEqual(get_model_name(policy.compactor_model), "gemini-3.1-flash-lite")

    def test_compactor_preserves_user_assistant_and_replaces_internal_events(self) -> None:
        policy, model = scripted_trajectory_policy(
            ["High-signal summary for turn 3.", "High-signal summary for turn 4."]
        )
        compactor = TrajectoryCompactor(policy=policy)
        events = [
            event(1, EventType.USER, "user 1", role="user"),
            event(1, EventType.TOOL, "tool args", name="pytest"),
            event(1, EventType.TOOL_OUTPUT, "long logs", role="tool"),
            event(1, EventType.ASSISTANT, "assistant 1", role="assistant"),
            event(2, EventType.USER, "user 2", role="user"),
            event(2, EventType.REASONING, "trace failure"),
            event(2, EventType.ASSISTANT, "assistant 2", role="assistant"),
            event(3, EventType.USER, "user 3", role="user"),
            event(3, EventType.REASONING, "reasoning 3"),
            event(3, EventType.ASSISTANT, "assistant 3", role="assistant"),
            event(4, EventType.USER, "user 4", role="user"),
            event(4, EventType.REASONING, "reasoning 4"),
            event(4, EventType.ASSISTANT, "assistant 4", role="assistant"),
        ]

        result = compactor.compact(events)

        self.assertEqual(result.compacted_turns, [3, 4])
        self.assertEqual(result.compacted_event_count, 2)
        visible_and_memory_contents = [
            item.payload["content"]
            for item in result.events
            if item.type in {EventType.USER, EventType.ASSISTANT} or item.payload.get("kind") == "trajectory_memory"
        ]
        self.assertEqual(
            visible_and_memory_contents,
            [
                "user 1",
                "assistant 1",
                "user 2",
                "assistant 2",
                "user 3",
                trajectory_memory_message(result.turn_syntheses[0]),
                "assistant 3",
                "user 4",
                trajectory_memory_message(result.turn_syntheses[1]),
                "assistant 4",
            ],
        )
        self.assertEqual(
            [(item.turn, item.synthesis) for item in result.turn_syntheses],
            [
                (3, "High-signal summary for turn 3."),
                (4, "High-signal summary for turn 4."),
            ],
        )
        self.assertEqual([item.payload.get("kind") for item in result.events if item.payload.get("kind") == "trajectory_memory"], ["trajectory_memory", "trajectory_memory"])
        self.assertFalse(
            any(
                (
                    item.type in {EventType.TOOL, EventType.TOOL_OUTPUT}
                    or (item.type == EventType.REASONING and item.payload.get("kind") != "trajectory_memory")
                )
                and item.turn in {3, 4}
                for item in result.events
            )
        )
        self.assertTrue(any(item.type == EventType.REASONING and item.turn == 2 for item in result.events))
        self.assertIn('TURN 3', model.calls[0][1])
        self.assertIn('Historical user requests:\nuser 3', model.calls[0][1])
        self.assertNotIn("assistant_message", model.calls[0][1])
        self.assertIn("assistant 3", model.calls[0][1])
        self.assertEqual(len(model.calls), 2)
        self.assertNotIn("reasoning 4", model.calls[0][1])
        self.assertNotIn("reasoning 3", model.calls[1][1])
        self.assertEqual(result.token_usage["call_count"], 2)
        self.assertEqual(result.token_usage["estimated_calls"], 2)

    def test_compactor_compacts_intermediate_assistant_events_but_keeps_final_assistant(self) -> None:
        policy, _ = scripted_trajectory_policy(
            "High-signal summary for turn 1.",
            trigger_every_turns=1,
        )
        compactor = TrajectoryCompactor(policy=policy)
        events = [
            event(1, EventType.USER, "user 1", role="user"),
            event(1, EventType.ASSISTANT, "internal draft", role="assistant"),
            event(1, EventType.ASSISTANT, "assistant 1", role="assistant"),
        ]

        result = compactor.compact(events)

        turn_one_contents = [item.payload["content"] for item in result.events if item.turn == 1]
        self.assertEqual(
            turn_one_contents,
            [
                "user 1",
                trajectory_memory_message(result.turn_syntheses[0]),
                "assistant 1",
            ],
        )
        self.assertNotIn("internal draft", turn_one_contents)
        self.assertEqual(result.compacted_event_count, 1)

    def test_compactor_waits_until_full_window_exists(self) -> None:
        policy, _ = scripted_trajectory_policy(
            "High-signal summary for turn 1.",
            trigger_every_turns=1,
        )
        compactor = TrajectoryCompactor(policy=policy)
        events = [
            event(1, EventType.USER, "user 1", role="user"),
            event(1, EventType.TOOL, "tool args", name="pytest"),
            event(1, EventType.ASSISTANT, "assistant 1", role="assistant"),
        ]

        self.assertTrue(compactor.should_compact(events))
        result = compactor.compact(events)
        self.assertEqual(result.compacted_turns, [1])

    def test_compactor_uses_latest_closed_batch(self) -> None:
        policy, _ = scripted_trajectory_policy(
            ["High-signal summary for turn 9.", "High-signal summary for turn 10."]
        )
        compactor = TrajectoryCompactor(policy=policy)
        events: list[SessionEvent] = []
        for turn in range(1, 11):
            events.extend(
                [
                    event(turn, EventType.USER, f"user {turn}", role="user"),
                    event(turn, EventType.REASONING, f"reasoning {turn}"),
                    event(turn, EventType.ASSISTANT, f"assistant {turn}", role="assistant"),
                ]
            )

        self.assertTrue(compactor.should_compact(events))
        result = compactor.compact(events)

        self.assertEqual(result.compacted_turns, [9, 10])
        self.assertTrue(any(item.payload.get("kind") == "trajectory_memory" and item.turn == 9 for item in result.events))
        self.assertTrue(any(item.payload.get("kind") == "trajectory_memory" and item.turn == 10 for item in result.events))
        self.assertTrue(any(item.type == EventType.REASONING and item.turn == 7 for item in result.events))
        self.assertTrue(any(item.type == EventType.REASONING and item.turn == 8 for item in result.events))
        self.assertFalse(
            any(item.type == EventType.REASONING and item.turn == 9 and item.payload.get("kind") != "trajectory_memory" for item in result.events)
        )
        self.assertFalse(
            any(item.type == EventType.REASONING and item.turn == 10 and item.payload.get("kind") != "trajectory_memory" for item in result.events)
        )

    def test_compactor_compacts_cancelled_turn_and_inserts_memory_before_turn_end(self) -> None:
        policy, _ = scripted_trajectory_policy(
            "High-signal summary for turn 2.",
            trigger_every_turns=1,
        )
        compactor = TrajectoryCompactor(policy=policy)
        events = [
            event(1, EventType.USER, "user 1", role="user"),
            event(1, EventType.REASONING, "reasoning 1"),
            event(1, EventType.ASSISTANT, "assistant 1", role="assistant"),
            event(2, EventType.USER, "user 2", role="user"),
            event(2, EventType.REASONING, "reasoning 2"),
            SessionEvent(type=EventType.TURN_END, turn=2, payload={"status": "cancelled"}),
        ]

        result = compactor.compact(events)

        self.assertEqual(result.compacted_turns, [2])
        turn_two_events = [item for item in result.events if item.turn == 2]
        memory_index = next(index for index, item in enumerate(turn_two_events) if item.payload.get("kind") == "trajectory_memory")
        turn_end_index = next(index for index, item in enumerate(turn_two_events) if item.type == EventType.TURN_END)
        self.assertLess(memory_index, turn_end_index)

    def test_compactor_inserts_memory_before_turn_end_when_no_assistant_exists(self) -> None:
        policy, _ = scripted_trajectory_policy(
            "High-signal summary for turn 1.",
            trigger_every_turns=1,
        )
        compactor = TrajectoryCompactor(policy=policy)
        events = [
            event(1, EventType.USER, "user 1", role="user"),
            event(1, EventType.REASONING, "reasoning 1"),
            SessionEvent(type=EventType.TURN_END, turn=1, payload={}),
        ]

        result = compactor.compact(events)

        turn_one_events = [item for item in result.events if item.turn == 1]
        memory_index = next(index for index, item in enumerate(turn_one_events) if item.payload.get("kind") == "trajectory_memory")
        turn_end_index = next(index for index, item in enumerate(turn_one_events) if item.type == EventType.TURN_END)
        self.assertLess(memory_index, turn_end_index)

    def test_coordinator_applies_pending_compaction_during_agent_preparation(self) -> None:
        with TemporaryDirectory() as directory:
            manager = SessionManager(session_id="s1", root=Path(directory))
            manager.append(
                [
                    event(1, EventType.USER, "user 1", role="user"),
                    event(1, EventType.TOOL, "tool args", name="pytest"),
                    event(1, EventType.ASSISTANT, "assistant 1", role="assistant"),
                    event(2, EventType.USER, "user 2", role="user"),
                    event(2, EventType.REASONING, "reasoning"),
                    event(2, EventType.ASSISTANT, "assistant 2", role="assistant"),
                    event(3, EventType.USER, "user 3", role="user"),
                    event(3, EventType.REASONING, "reasoning 3"),
                    event(3, EventType.ASSISTANT, "assistant 3", role="assistant"),
                    event(4, EventType.USER, "user 4", role="user"),
                    event(4, EventType.REASONING, "reasoning 4"),
                    event(4, EventType.ASSISTANT, "assistant 4", role="assistant"),
                ]
            )
            started = Event()
            release = Event()
            blocking_model = BlockingModel(
                ["Summary 3.", "Summary 4."],
                started=started,
                release=release,
            )
            coordinator = TrajectoryCompactionCoordinator(
                manager,
                TrajectoryCompactor(
                    policy=TrajectoryCompactionPolicy(trigger_every_turns=2, compactor_model=blocking_model),
                ),
                telemetry_store=TelemetryStore(Path(directory) / "telemetry.jsonl"),
                repo_root=Path(directory),
            )

            status = coordinator.request_compaction()
            self.assertEqual(status, "started")
            self.assertTrue(started.wait(timeout=1))

            manager.append(
                [
                    event(5, EventType.USER, "user 5", role="user"),
                    event(5, EventType.ASSISTANT, "assistant 5", role="assistant"),
                ]
            )
            release.set()

            deadline = datetime.now(timezone.utc) + timedelta(seconds=2)
            while coordinator.is_running() and datetime.now(timezone.utc) < deadline:
                sleep(0.01)

            curated = manager.read_curated()
            self.assertFalse(any(item.payload.get("kind") == "trajectory_memory" for item in curated))

            status = coordinator.prepare_for_agent()
            self.assertEqual(status, "applied")

            curated = manager.read_curated()
            self.assertTrue(any(item.payload.get("kind") == "trajectory_memory" for item in curated))
            user3_index = next(index for index, item in enumerate(curated) if item.payload["content"] == "user 3")
            assistant3_index = next(index for index, item in enumerate(curated) if item.payload["content"] == "assistant 3")
            self.assertEqual(curated[user3_index + 1].payload.get("kind"), "trajectory_memory")
            self.assertEqual(user3_index + 2, assistant3_index)
            self.assertEqual(curated[-2].payload["content"], "user 5")
            self.assertEqual(curated[-1].payload["content"], "assistant 5")

    def test_prepare_for_agent_blocks_until_running_compaction_finishes(self) -> None:
        with TemporaryDirectory() as directory:
            manager = SessionManager(session_id="s1", root=Path(directory))
            manager.append(
                [
                    event(1, EventType.USER, "user 1", role="user"),
                    event(1, EventType.TOOL, "tool args", name="pytest"),
                    event(1, EventType.ASSISTANT, "assistant 1", role="assistant"),
                    event(2, EventType.USER, "user 2", role="user"),
                    event(2, EventType.REASONING, "reasoning 2"),
                    event(2, EventType.ASSISTANT, "assistant 2", role="assistant"),
                    event(3, EventType.USER, "user 3", role="user"),
                    event(3, EventType.REASONING, "reasoning 3"),
                    event(3, EventType.ASSISTANT, "assistant 3", role="assistant"),
                    event(4, EventType.USER, "user 4", role="user"),
                    event(4, EventType.REASONING, "reasoning 4"),
                    event(4, EventType.ASSISTANT, "assistant 4", role="assistant"),
                ]
            )
            started = Event()
            release = Event()
            blocking_model = BlockingModel(
                ["Summary 3.", "Summary 4."],
                started=started,
                release=release,
            )
            coordinator = TrajectoryCompactionCoordinator(
                manager,
                TrajectoryCompactor(
                    policy=TrajectoryCompactionPolicy(trigger_every_turns=2, compactor_model=blocking_model),
                ),
                telemetry_store=TelemetryStore(Path(directory) / "telemetry.jsonl"),
                repo_root=Path(directory),
            )

            self.assertEqual(coordinator.request_compaction(), "started")
            self.assertTrue(started.wait(timeout=1))

            release.set()
            status = coordinator.prepare_for_agent()

            self.assertEqual(status, "applied")
            curated = manager.read_curated()
            self.assertTrue(any(item.payload.get("kind") == "trajectory_memory" for item in curated))

    def test_prepare_for_agent_loads_pending_compaction_from_disk(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manager = SessionManager(session_id="s1", root=root)
            manager.append(
                [
                    event(1, EventType.USER, "user 1", role="user"),
                    event(1, EventType.TOOL, "tool args", name="pytest"),
                    event(1, EventType.ASSISTANT, "assistant 1", role="assistant"),
                ]
            )
            policy, _ = scripted_trajectory_policy(
                "High-signal summary for turn 1.",
                trigger_every_turns=1,
            )
            first = TrajectoryCompactionCoordinator(
                manager,
                TrajectoryCompactor(policy=policy),
                telemetry_store=TelemetryStore(root / "telemetry.jsonl"),
                repo_root=root,
            )

            self.assertEqual(first.request_compaction(), "started")
            deadline = datetime.now(timezone.utc) + timedelta(seconds=2)
            while first.is_running() and datetime.now(timezone.utc) < deadline:
                sleep(0.01)

            pending_path = root / "pending_trajectories" / "s1.json"
            self.assertTrue(pending_path.exists())
            pending = json.loads(pending_path.read_text())
            breakdown = pending["payload"]["context_tokens"]
            self.assertGreater(breakdown["compacted_tokens"], 0)
            self.assertGreater(breakdown["injected_tokens"], 0)
            self.assertEqual(breakdown["saved_tokens"], breakdown["before_tokens"] - breakdown["after_tokens"])
            self.assertFalse(any(item.payload.get("kind") == "trajectory_memory" for item in manager.read_curated()))

            second = TrajectoryCompactionCoordinator(
                manager,
                TrajectoryCompactor(policy=policy),
                telemetry_store=TelemetryStore(root / "telemetry-2.jsonl"),
                repo_root=root,
            )

            self.assertEqual(second.prepare_for_agent(), "applied")
            self.assertFalse(pending_path.exists())
            curated = manager.read_curated()
            self.assertTrue(any(item.payload.get("kind") == "trajectory_memory" for item in curated))
            logged = [json.loads(line) for line in (root / "telemetry-2.jsonl").read_text().splitlines()][-1]["payload"]
            self.assertEqual(logged["context_tokens"], breakdown)
            self.assertEqual(logged["token_usage"]["input_tokens"], pending["token_usage"]["input_tokens"])
            self.assertEqual(logged["token_usage"]["estimated_calls"], 1)

    def test_format_turn_interval_coalesces_ranges(self) -> None:
        self.assertEqual(format_turn_interval([1, 2, 3, 5, 7, 8]), "[1-3, 5, 7-8]")

    def test_markdown_excerpts_survive_pending_storage_and_session_restore(self) -> None:
        excerpt = '\tif name == "a\\tb":\n\t\treturn {"key": value}  # exact spacing'
        memory = f'I rejected normalization after inspecting parser.py.\n\n```python\n{excerpt}\n```'
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manager = SessionManager(session_id="literal", root=root)
            source = [
                event(1, EventType.USER, "Repair the parser."),
                event(1, EventType.TOOL_OUTPUT, excerpt, role="tool"),
                event(1, EventType.ASSISTANT, "Repaired; no tests run."),
                SessionEvent(type=EventType.TURN_END, turn=1, payload={}),
            ]
            manager.append(source)
            raw_before = manager.dump_path.read_bytes()
            policy, _ = scripted_trajectory_policy(memory, trigger_every_turns=1)
            coordinator = TrajectoryCompactionCoordinator(
                manager, TrajectoryCompactor(policy=policy), repo_root=root,
                telemetry_store=TelemetryStore(root / "telemetry.jsonl"),
            )
            coordinator._run_compaction(source, 1)
            reloaded = TrajectoryCompactionCoordinator(
                manager, TrajectoryCompactor(policy=policy), repo_root=root,
                telemetry_store=TelemetryStore(root / "telemetry.jsonl"),
            )
            self.assertEqual(reloaded.prepare_for_agent(), "applied")
            restored = manager.load_curated_messages()
            restored_memory = restored[1].content[0]["thinking"]
            self.assertIn(memory, restored_memory)
            self.assertIn(excerpt, restored_memory)
            self.assertNotIn("Live edge:", restored_memory)
            self.assertEqual(manager.dump_path.read_bytes(), raw_before)
            self.assertFalse(any(item.payload.get("kind") == "trajectory_memory"
                                 for item in manager.read_display_history()))

    def test_failed_or_incomplete_synthesis_preserves_both_session_files(self) -> None:
        responses = [
            SimpleNamespace(content="   "),
            SimpleNamespace(content=None),
            SimpleNamespace(content=[{"type": "reasoning", "text": "Internal thoughts only"}]),
            SimpleNamespace(content=[{"type": "refusal", "refusal": "Cannot comply"}]),
            *[SimpleNamespace(content="Partial memory", response_metadata={"finish_reason": reason})
              for reason in ("MAX_TOKENS", "length", "SAFETY", "RECITATION", "content_filter")],
            SimpleNamespace(content="Partial memory", response_metadata={"stop_reason": "max_tokens"}),
        ]
        for response in responses:
            with self.subTest(response=response), TemporaryDirectory() as directory:
                root = Path(directory)
                manager = SessionManager(session_id="failure", root=root)
                source = [event(1, EventType.USER, "Task"), event(1, EventType.REASONING, "Evidence")]
                manager.append(source)
                before = (manager.dump_path.read_bytes(), manager.curated_path.read_bytes())
                policy = TrajectoryCompactionPolicy(
                    trigger_every_turns=1, compactor_model=SimpleNamespace(invoke=lambda _: response),
                )
                telemetry_path = root / "telemetry.jsonl"
                coordinator = TrajectoryCompactionCoordinator(
                    manager, TrajectoryCompactor(policy=policy), repo_root=root,
                    telemetry_store=TelemetryStore(telemetry_path),
                )
                coordinator._run_compaction(source, 1)
                self.assertEqual(coordinator.prepare_for_agent(), "not_ready")
                self.assertEqual((manager.dump_path.read_bytes(), manager.curated_path.read_bytes()), before)
                self.assertFalse((root / "pending_trajectories/failure.json").exists())
                logged = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
                self.assertEqual(logged[-1]["payload"]["exception_type"], "ValueError")

    def test_only_visible_text_blocks_become_memory(self) -> None:
        response = SimpleNamespace(content=[
            {"type": "reasoning", "text": "Do not persist these thoughts"},
            {"type": "text", "text": "I found a literal condition."},
            {"type": "text", "text": "```python\n\treturn value\n```"},
        ])
        policy = TrajectoryCompactionPolicy(
            trigger_every_turns=1, compactor_model=SimpleNamespace(invoke=lambda _: response),
        )
        result = TrajectoryCompactor(policy=policy).compact([event(1, EventType.REASONING, "Source")])
        self.assertEqual(result.memory_document, "I found a literal condition.\n```python\n\treturn value\n```")

    def test_input_preserves_decoded_evidence_and_live_user_corrections(self) -> None:
        excerpt = '\treturn {"name": "a\\tb"}\n\t# retain tabs and literal escapes'
        policy, model = scripted_trajectory_policy("I retained the evidence.", trigger_every_turns=1)
        result = TrajectoryCompactor(policy=policy).compact([
            event(1, EventType.USER, "Change the parser."),
            event(1, EventType.TOOL_OUTPUT, excerpt, role="tool", tool_call_id="read-1"),
            event(1, EventType.USER, "Do not normalize the tabs."),
            event(1, EventType.ASSISTANT, "Unverified final claim."),
        ])
        request = model.calls[0][1]
        self.assertIn(excerpt, request)
        self.assertIn('"tool_call_id": "read-1"', request)
        self.assertIn("Change the parser.\n\nDo not normalize the tabs.", request)
        self.assertIn("claims, not independent evidence", request)
        self.assertIn("Unverified final claim.", request)
        self.assertIn("Unverified final claim.", [item.payload.get("content") for item in result.events])

    def test_synthesis_wrapper_preserves_verbatim_content(self) -> None:
        content = 'I found the mismatch.\n\n```diff\n-\tvalue = cfg["api-key"]\n+\tvalue = cfg["api_key"]\n```'
        self.assertEqual(
            trajectory_memory_message(TurnTrajectorySynthesis(turn=1, synthesis=content)),
            f"[TRAJECTORY SYNTHESIS]\n\n{content}\n\n[END TRAJECTORY SYNTHESIS]",
        )

    def test_existing_memories_need_no_migration_or_resynthesis(self) -> None:
        policy, model = scripted_trajectory_policy("Should not be called", trigger_every_turns=1)
        existing = [
            event(1, EventType.USER, "Previous task"),
            event(1, EventType.REASONING, "[TRAJECTORY MEMORY]\nOld synthesis\nLive edge:\nOld cue",
                  kind="trajectory_memory"),
            event(1, EventType.ASSISTANT, "Previous answer"),
        ]
        result = TrajectoryCompactor(policy=policy).compact(existing)
        self.assertEqual(result.events, existing)
        self.assertEqual(result.compacted_turns, [])
        self.assertEqual(model.calls, [])

    def test_real_token_guard_keeps_dense_turn_but_compacts_noisy_turn(self) -> None:
        policy, _ = scripted_trajectory_policy(["Expanded note. " * 100, "I found the failing condition."])
        source = [
            event(1, EventType.USER, "First task"),
            event(1, EventType.REASONING, "One useful finding."),
            event(1, EventType.ASSISTANT, "First answer"),
            event(2, EventType.USER, "Second task"),
            event(2, EventType.REASONING, "Repeated noise. " * 1000),
            event(2, EventType.ASSISTANT, "Second answer"),
        ]
        counter = TokenCounter()
        result = TrajectoryCompactor(policy=policy, token_counter=counter).compact(source)
        self.assertEqual(result.compacted_turns, [2])
        self.assertEqual([item for item in result.events if item.turn == 1], source[:3])
        self.assertEqual(result.compacted_event_count, 1)
        self.assertEqual(result.token_usage["call_count"], 2)
        self.assertLess(counter.count_context(result.events), counter.count_context(source))

    def test_no_savings_skips_publication_and_records_usage(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manager = SessionManager(session_id="dense", root=root)
            source = [event(1, EventType.USER, "Task"), event(1, EventType.REASONING, "Finding")]
            manager.append(source)
            before = manager.curated_path.read_bytes()
            policy, _ = scripted_trajectory_policy("Long memory. " * 100, trigger_every_turns=1)
            telemetry_path = root / "telemetry.jsonl"
            coordinator = TrajectoryCompactionCoordinator(
                manager, TrajectoryCompactor(policy=policy, token_counter=TokenCounter()), repo_root=root,
                telemetry_store=TelemetryStore(telemetry_path),
            )
            coordinator._run_compaction(source, 1)
            self.assertEqual(coordinator.prepare_for_agent(), "not_ready")
            self.assertEqual(manager.curated_path.read_bytes(), before)
            logged = [json.loads(line) for line in telemetry_path.read_text().splitlines()][-1]["payload"]
            self.assertEqual(logged["reason"], "no_context_reduction")
            self.assertEqual(logged["token_usage"]["call_count"], 1)

    def test_compactor_rejects_empty_response_without_rewriting_batch(self) -> None:
        policy, _ = scripted_trajectory_policy(["Valid first note.", "   "])
        compactor = TrajectoryCompactor(policy=policy)
        events = [
            event(1, EventType.USER, "user 1", role="user"),
            event(1, EventType.TOOL, "tool args", name="pytest"),
            event(1, EventType.ASSISTANT, "assistant 1", role="assistant"),
            event(2, EventType.USER, "user 2", role="user"),
            event(2, EventType.REASONING, "trace failure"),
            event(2, EventType.ASSISTANT, "assistant 2", role="assistant"),
            event(3, EventType.USER, "user 3", role="user"),
            event(3, EventType.REASONING, "reasoning 3"),
            event(3, EventType.ASSISTANT, "assistant 3", role="assistant"),
            event(4, EventType.USER, "user 4", role="user"),
            event(4, EventType.REASONING, "reasoning 4"),
            event(4, EventType.ASSISTANT, "assistant 4", role="assistant"),
        ]

        original = [item.to_json_dict() for item in events]
        with self.assertRaisesRegex(ValueError, "Trajectory synthesis is empty"):
            compactor.compact(events)
        self.assertEqual([item.to_json_dict() for item in events], original)
