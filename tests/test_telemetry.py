from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import dataclass
from typing import Any
from unittest import TestCase

from langchain.agents.middleware import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from core.compaction.token_counter import TokenCounter
from core.middleware.telemetry import TelemetryMiddleware, _exception_payload, _model_name, _response_usage
from core.telemetry.events import TelemetryEvent
from core.telemetry.store import TelemetryStore, telemetry_session_path


class FixedTokenCounter(TokenCounter):
    def count_text(self, text: str) -> int:
        return 123


@dataclass(frozen=True)
class FakeRequest:
    model: Any
    messages: list[Any]


@dataclass(frozen=True)
class FakeToolRequest:
    tool_call: dict[str, Any]
    tool: Any = None
    state: Any = None
    runtime: Any = None


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

    def test_response_usage_sums_message_usage_metadata(self) -> None:
        response = ModelResponse(
            result=[
                AIMessage(content="a", usage_metadata={"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}),
                AIMessage(content="b", usage_metadata={"input_tokens": 4, "output_tokens": 1, "total_tokens": 5}),
            ]
        )

        self.assertEqual(
            _response_usage(response),
            {"input_tokens": 6, "output_tokens": 4, "total_tokens": 10},
        )

    def test_response_usage_sums_nested_reasoning_details(self) -> None:
        response = ModelResponse(
            result=[
                AIMessage(
                    content="a",
                    usage_metadata={
                        "input_tokens": 2,
                        "output_tokens": 3,
                        "total_tokens": 5,
                        "output_token_details": {"reasoning": 7},
                    },
                ),
                AIMessage(
                    content="b",
                    usage_metadata={
                        "input_tokens": 1,
                        "output_tokens": 4,
                        "total_tokens": 5,
                        "output_token_details": {"reasoning": 5},
                    },
                ),
            ]
        )

        self.assertEqual(
            _response_usage(response),
            {
                "input_tokens": 3,
                "output_tokens": 7,
                "total_tokens": 10,
                "output_token_details": {"reasoning": 12},
            },
        )

    def test_telemetry_middleware_records_token_counts_and_usage(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.jsonl"
            middleware = TelemetryMiddleware(TelemetryStore(path), token_counter=FixedTokenCounter())
            request = FakeRequest(model="fake-model", messages=[HumanMessage(content="hello")])

            response = middleware.wrap_model_call(
                request,  # type: ignore[arg-type]
                lambda _: ModelResponse(
                    result=[
                        AIMessage(
                            content="hi",
                            usage_metadata={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
                        )
                    ]
                ),
            )

            self.assertEqual(response.result[0].content, "hi")
            text = path.read_text(encoding="utf-8")
            self.assertIn('"input_tokens_estimate": 123', text)
            self.assertIn('"usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}', text)

    def test_telemetry_middleware_records_structured_model_exception(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.jsonl"
            middleware = TelemetryMiddleware(TelemetryStore(path), token_counter=FixedTokenCounter())
            request = FakeRequest(model="fake-model", messages=[HumanMessage(content="hello")])

            with self.assertRaises(RuntimeError):
                middleware.wrap_model_call(  # type: ignore[arg-type]
                    request,
                    lambda _: (_ for _ in ()).throw(RuntimeError("429 RESOURCE_EXHAUSTED Please retry in 11s")),
                )

            text = path.read_text(encoding="utf-8")
            self.assertIn('"name": "model.error"', text)
            self.assertIn('"error_type": "RuntimeError"', text)
            self.assertIn('"provider_status": "RESOURCE_EXHAUSTED"', text)
            self.assertIn('"retry_delay_seconds": 11', text)

    def test_exception_payload_extracts_chained_status_code(self) -> None:
        cause = ValueError("provider failed")
        cause.status_code = 429  # type: ignore[attr-defined]
        exc = RuntimeError("wrapper")
        exc.__cause__ = cause

        payload = _exception_payload(exc)

        self.assertEqual(payload["status_code"], 429)
        self.assertEqual(payload["chain"][1]["type"], "ValueError")

    def test_telemetry_middleware_records_tool_metadata(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.jsonl"
            middleware = TelemetryMiddleware(TelemetryStore(path), token_counter=FixedTokenCounter())
            request = FakeToolRequest(
                tool_call={
                    "id": "call-1",
                    "name": "read_file",
                    "args": {"path": "/tmp/example.txt"},
                }
            )

            result = middleware.wrap_tool_call(
                request,
                lambda _: ToolMessage(content="tool output", tool_call_id="call-1"),
            )

            self.assertEqual(result.content, "tool output")
            text = path.read_text(encoding="utf-8")
            self.assertIn('"name": "tool.start"', text)
            self.assertIn('"name": "tool.end"', text)
            self.assertIn('"tool_name": "read_file"', text)
            self.assertIn('"tool_call_id": "call-1"', text)
            self.assertIn('"content_chars": 11', text)

    def test_telemetry_middleware_records_tool_exceptions(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.jsonl"
            middleware = TelemetryMiddleware(TelemetryStore(path), token_counter=FixedTokenCounter())
            request = FakeToolRequest(tool_call={"id": "call-1", "name": "shell", "args": {"cmd": "bad"}})

            with self.assertRaises(RuntimeError):
                middleware.wrap_tool_call(
                    request,
                    lambda _: (_ for _ in ()).throw(RuntimeError("tool failed")),
                )

            text = path.read_text(encoding="utf-8")
            self.assertIn('"name": "tool.error"', text)
            self.assertIn('"tool_name": "shell"', text)
            self.assertIn('"error_type": "RuntimeError"', text)
