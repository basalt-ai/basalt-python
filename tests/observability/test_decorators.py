"""Tests for observability decorators."""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from unittest import mock

from basalt.observability import semconv
from basalt.observability.decorators import (
    trace_event as trace_event_decorator,
)
from basalt.observability.decorators import (
    trace_function as trace_function_decorator,
)
from basalt.observability.decorators import (
    trace_generation as trace_generation_decorator,
)
from basalt.observability.decorators import (
    trace_retrieval as trace_retrieval_decorator,
)
from basalt.observability.decorators import (
    trace_span as trace_span_decorator,
)
from basalt.observability.decorators import (
    trace_tool as trace_tool_decorator,
)
from tests.observability.utils import get_exporter


class DecoratorTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.exporter = get_exporter()
        self.exporter.clear()

    def _single_span(self):
        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        return spans[0]

    def test_trace_span_decorator_records_io_and_evaluators(self):
        @trace_span_decorator(
            name="workflow.compute",
            variables=lambda bound: {"user": bound.arguments["user"]},
            evaluators=["quality"],
        )
        def compute(user: str, value: int) -> dict[str, int]:
            return {"total": value * 2}

        compute("alice", 3)

        span = self._single_span()
        self.assertEqual(span.name, "workflow.compute")
        self.assertEqual(span.attributes[semconv.BasaltSpan.TYPE], "span")

        input_payload = json.loads(span.attributes[semconv.BasaltSpan.INPUT])
        self.assertEqual(input_payload["user"], "alice")

        output_payload = json.loads(span.attributes[semconv.BasaltSpan.OUTPUT])
        self.assertEqual(output_payload["total"], 6)

        variables_payload = json.loads(span.attributes[semconv.BasaltSpan.VARIABLES])
        self.assertEqual(variables_payload["user"], "alice")

        evaluators = span.attributes[semconv.BasaltSpan.EVALUATORS]
        self.assertIn("quality", evaluators)

    def test_trace_span_async_support(self):
        @trace_span_decorator(name="async.operation")
        async def sample_async(value: int) -> int:
            await asyncio.sleep(0)
            return value + 1

        asyncio.run(sample_async(3))

        span = self._single_span()
        self.assertEqual(span.name, "async.operation")
        output_payload = span.attributes.get(semconv.BasaltSpan.OUTPUT)
        self.assertIsNotNone(output_payload)

    def test_trace_generation_records_prompt_and_completion(self):
        @trace_generation_decorator(name="llm.generate")
        def generate(model: str, prompt: str):
            return {
                "choices": [{"message": {"content": "done"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            }

        with mock.patch.dict(os.environ, {"TRACELOOP_TRACE_CONTENT": "true"}, clear=False):
            generate(model="gpt-4", prompt="Hello?")

        span = self._single_span()
        self.assertEqual(span.name, "llm.generate")
        self.assertEqual(span.attributes[semconv.BasaltSpan.TYPE], "generation")
        self.assertEqual(span.attributes[semconv.GenAI.REQUEST_MODEL], "gpt-4")

        input_messages = json.loads(span.attributes[semconv.GenAI.INPUT_MESSAGES])
        self.assertEqual(input_messages[0]["parts"][0]["content"], "Hello?")

        output_messages = json.loads(span.attributes[semconv.GenAI.OUTPUT_MESSAGES])
        self.assertEqual(output_messages[0]["parts"][0]["content"], "done")

        self.assertEqual(span.attributes[semconv.GenAI.USAGE_INPUT_TOKENS], 5)
        self.assertEqual(span.attributes[semconv.GenAI.USAGE_OUTPUT_TOKENS], 2)

    def test_trace_tool_sets_name_and_payload(self):
        @trace_tool_decorator(name="tool.invoke", tool_name=lambda bound: bound.arguments["name"])
        def invoke(name: str, payload: dict[str, str]) -> dict[str, bool]:
            return {"ok": True}

        invoke("search", {"query": "hello"})

        span = self._single_span()
        self.assertEqual(span.name, "tool.invoke")
        self.assertEqual(span.attributes[semconv.BasaltSpan.TYPE], "tool")
        self.assertEqual(span.attributes[semconv.BasaltTool.NAME], "search")
        tool_input = json.loads(span.attributes[semconv.BasaltTool.INPUT])
        self.assertEqual(tool_input["query"], "hello")

    def test_trace_function_sets_metadata(self):
        @trace_function_decorator(
            name="compute.embed",
            function_name=lambda bound: bound.arguments["task"],
            stage=lambda bound: "postprocess",
        )
        def embed(task: str, values: list[int]) -> list[int]:
            return [val * 3 for val in values]

        embed("scoring", [1, 2])

        span = self._single_span()
        self.assertEqual(span.attributes[semconv.BasaltSpan.TYPE], "function")
        self.assertEqual(span.attributes[semconv.BasaltFunction.NAME], "scoring")
        self.assertEqual(span.attributes[semconv.BasaltFunction.STAGE], "postprocess")

    def test_trace_event_sets_type(self):
        @trace_event_decorator(name="event.notify", event_type="user-action")
        def notify(payload: dict[str, str]) -> dict[str, str]:
            return {"status": "sent"}

        notify({"message": "hi"})

        span = self._single_span()
        self.assertEqual(span.attributes[semconv.BasaltSpan.TYPE], "event")
        self.assertEqual(span.attributes[semconv.BasaltEvent.TYPE], "user-action")
        event_input = json.loads(span.attributes[semconv.BasaltEvent.PAYLOAD])
        self.assertEqual(event_input["message"], "hi")

    def test_trace_retrieval_captures_query(self):
        @trace_retrieval_decorator(name="retrieval.search")
        def search(query: str) -> list[str]:
            return ["doc1", "doc2"]

        search("basalt")

        span = self._single_span()
        self.assertEqual(span.attributes[semconv.BasaltSpan.TYPE], "retrieval")
        self.assertEqual(span.attributes[semconv.BasaltRetrieval.QUERY], "basalt")
