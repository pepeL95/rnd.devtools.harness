from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from core.compaction.compactor import Compactor
from core.compaction.policy import CompactionPolicy
from core.compaction.serialization import memory_restore_message
from core.compaction.window import split_compaction_window
from core.middleware.compaction import CompactionMiddleware
from core.session.events import EventType, SessionEvent
from core.session.session_manager import SessionManager


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


class CompactionTests(TestCase):
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
                "MEMORY DOCUMENT",
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
        self.assertIn("MEMORY DOCUMENT", result.events[0].payload["content"])
        self.assertEqual(result.events[1].payload["content"], "continue")

    def test_compactor_runs_revision_loop_until_approved(self) -> None:
        generator = ScriptedGenerator(
            [
                "EPISODE 1: setup\nTURNS: 1 to 1",
                "DRAFT MEMORY",
                "CRITIQUE SUMMARY\n  VIOLATIONS FOUND: 1\n  RECOMMENDED ACTION: revise targeted sections",
                "REVISED MEMORY",
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
        self.assertEqual(result.memory_document, "REVISED MEMORY")
        self.assertEqual(len(result.critiques), 2)

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
                        "CURATED MEMORY",
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

    def test_memory_restore_message_matches_injection_contract(self) -> None:
        content = memory_restore_message("DOC")

        self.assertTrue(content.startswith("[MEMORY RESTORE]"))
        self.assertIn("DOC", content)
        self.assertIn("[END MEMORY RESTORE]", content)
        self.assertTrue(content.endswith("Your task continues below."))

