from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from core.telemetry.events import TelemetryEvent
from core.telemetry.store import TelemetryStore, telemetry_session_path
from core.middleware.telemetry import _model_name


class TelemetryTests(TestCase):
    def test_telemetry_session_path_uses_date_partition(self) -> None:
        path = telemetry_session_path(
            "abc",
            root=Path("/tmp/telemetry"),
            timestamp=datetime(2026, 5, 23, tzinfo=timezone.utc),
        )

        self.assertEqual(path, Path("/tmp/telemetry/2026/05/23/abc.jsonl"))

    def test_telemetry_store_writes_jsonl(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.jsonl"
            store = TelemetryStore(path)

            store.record(TelemetryEvent(name="agent.start", payload={"messages": 1}))

            text = path.read_text(encoding="utf-8")
            self.assertIn('"name": "agent.start"', text)
            self.assertIn('"messages": 1', text)

    def test_model_name_avoids_full_model_repr(self) -> None:
        class FakeModel:
            model = "gemini-test"

        self.assertEqual(_model_name(FakeModel()), "gemini-test")
