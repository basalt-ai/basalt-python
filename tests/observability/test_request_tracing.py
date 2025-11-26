import importlib
import types
import unittest
from collections.abc import Iterator

import pytest
from opentelemetry.trace import StatusCode

from basalt.observability import semconv
from basalt.observability.request_tracing import trace_async_request, trace_sync_request
from basalt.observability.spans import BasaltRequestSpan


class DummySpan:
    def __init__(self):
        self.variables = None
        self.attributes = {}
        self.exceptions = []
        self.status = None
        self._io_payload = {"input": None, "output": None, "variables": None}

    def set_io(self, *, input_payload=None, output_payload=None, variables=None):
        if variables is not None:
            self.variables = variables
            self._io_payload["variables"] = variables

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def record_exception(self, exc):
        self.exceptions.append(exc)

    def set_status(self, status):
        self.status = status


class DummyObserve:
    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.entered = []

    def __call__(self, *, name, metadata):
        self.entered.append({"name": name, "metadata": metadata})
        return _DummyContext(self)

    def input(self, payload):
        self.inputs.append(payload)

    def output(self, payload):
        self.outputs.append(payload)


class _DummyContext:
    def __init__(self, observe):
        self.observe = observe
        self.span = DummySpan()

    def __enter__(self):
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class RecordingSpan(BasaltRequestSpan):
    __slots__ = ("finalize_calls",)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.finalize_calls: list[dict] = []

    def finalize(self, span, *, duration_s, status_code, error=None):
        self.finalize_calls.append(
            {"span": span, "duration_s": duration_s, "status_code": status_code, "error": error}
        )
        return super().finalize(span, duration_s=duration_s, status_code=status_code, error=error)


def _patch_perf_counter(monkeypatch, values: list[float]) -> None:
    iterator: Iterator[float] = iter(values)
    module = importlib.import_module("basalt.observability.request_tracing")
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(iterator))


class TraceSyncRequestTests(unittest.TestCase):
    def setUp(self):
        self.monkeypatch = pytest.MonkeyPatch()

    def tearDown(self):
        self.monkeypatch.undo()

    def test_trace_sync_request_success(self):
        dummy_observe = DummyObserve()
        self.monkeypatch.setattr("basalt.observability.request_tracing.observe", dummy_observe)
        _patch_perf_counter(self.monkeypatch, [10.0, 10.25])

        span_data = RecordingSpan(
            client="prompts",
            operation="create",
            method="post",
            url="https://example.test/prompts",
            variables={"foo": "bar"},
        )

        response = types.SimpleNamespace(status_code=201)
        result = trace_sync_request(span_data, lambda: response)

        self.assertIs(result, response)
        self.assertEqual(dummy_observe.inputs, [{"method": "post", "url": "https://example.test/prompts"}])
        self.assertEqual(dummy_observe.outputs, [{"status_code": 201}])

        metadata = dummy_observe.entered[0]["metadata"]
        self.assertEqual(metadata[semconv.HTTP.METHOD], "POST")
        self.assertEqual(metadata[semconv.HTTP.URL], "https://example.test/prompts")
        self.assertEqual(metadata[semconv.BasaltAPI.CLIENT], "prompts")

        finalize_call = span_data.finalize_calls[0]
        self.assertEqual(finalize_call["status_code"], 201)
        self.assertIsNone(finalize_call["error"])
        self.assertAlmostEqual(finalize_call["duration_s"], 0.25, places=2)

        dummy_span = finalize_call["span"]
        self.assertEqual(dummy_span.variables, {"foo": "bar"})
        self.assertTrue(dummy_span.attributes[semconv.BasaltRequest.SUCCESS])
        self.assertEqual(dummy_span._status, StatusCode.OK)


class TraceAsyncRequestTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.monkeypatch = pytest.MonkeyPatch()

    def tearDown(self):
        self.monkeypatch.undo()

    async def test_trace_async_request_records_error(self):
        dummy_observe = DummyObserve()
        self.monkeypatch.setattr("basalt.observability.request_tracing.observe", dummy_observe)
        _patch_perf_counter(self.monkeypatch, [3.0, 3.1])

        class FailingError(RuntimeError):
            def __init__(self, message, status_code):
                super().__init__(message)
                self.status_code = status_code

        span_data = RecordingSpan(
            client="datasets",
            operation="fetch",
            method="get",
            url="https://example.test/datasets/123",
        )

        async def failing_request():
            raise FailingError("boom", 503)

        with self.assertRaises(FailingError):
            await trace_async_request(span_data, failing_request)

        self.assertEqual(dummy_observe.inputs, [{"method": "get", "url": "https://example.test/datasets/123"}])
        self.assertEqual(dummy_observe.outputs, [{"error": "boom", "status_code": 503}])

        finalize_call = span_data.finalize_calls[0]
        self.assertEqual(finalize_call["status_code"], 503)
        self.assertIsInstance(finalize_call["error"], FailingError)
        self.assertAlmostEqual(finalize_call["duration_s"], 0.1, places=2)

        dummy_span = finalize_call["span"]
        self.assertFalse(dummy_span.attributes[semconv.BasaltRequest.SUCCESS])
        self.assertEqual(dummy_span._status, StatusCode.ERROR)
