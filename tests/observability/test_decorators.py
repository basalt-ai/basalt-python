"""Tests for observability decorators."""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from unittest import mock

from basalt.observability import semconv
from basalt.observability.decorators import trace_http, trace_llm, trace_operation
from tests.observability.utils import get_exporter


class DecoratorTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.exporter = get_exporter()
        self.exporter.clear()

    def test_trace_operation_records_attributes(self):
        @trace_operation(name="test.operation", attributes=lambda value: {"input": value}, capture_io=True)
        def sample(value: int) -> int:
            return value * 2

        result = sample(5)
        self.assertEqual(result, 10)

        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self.assertEqual(span.name, "test.operation")
        self.assertEqual(span.attributes["input"], 5)
        self.assertEqual(span.attributes[semconv.BasaltObserve.ARGS_COUNT], 1)

    def test_trace_operation_async(self):
        @trace_operation(name="async.operation")
        async def sample_async(value: int) -> int:
            await asyncio.sleep(0)
            return value + 1

        asyncio.run(sample_async(3))

        spans = self.exporter.get_finished_spans()
        self.assertTrue(any(span.name == "async.operation" for span in spans))

    def test_trace_llm_records_prompt_and_completion(self):
        @trace_llm(name="llm.generate")
        def generate(model: str, prompt: str):
            return {
                "choices": [{"message": {"content": "done"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            }

        with mock.patch.dict(os.environ, {"TRACELOOP_TRACE_CONTENT": "true"}, clear=False):
            generate(model="gpt-4", prompt="Hello?")

        span = self.exporter.get_finished_spans()[0]
        self.assertEqual(span.name, "llm.generate")

        # Check new GenAI semantic conventions
        self.assertEqual(span.attributes[semconv.GenAI.REQUEST_MODEL], "gpt-4")

        # Check that prompt is stored as structured messages
        input_messages = json.loads(span.attributes[semconv.GenAI.INPUT_MESSAGES])
        self.assertEqual(len(input_messages), 1)
        self.assertEqual(input_messages[0]["role"], "user")
        self.assertEqual(input_messages[0]["parts"][0]["content"], "Hello?")

        # Check that completion is stored as structured messages
        output_messages = json.loads(span.attributes[semconv.GenAI.OUTPUT_MESSAGES])
        self.assertEqual(len(output_messages), 1)
        self.assertEqual(output_messages[0]["role"], "assistant")
        self.assertEqual(output_messages[0]["parts"][0]["content"], "done")

        # Check token usage
        self.assertEqual(span.attributes[semconv.GenAI.USAGE_INPUT_TOKENS], 5)
        self.assertEqual(span.attributes[semconv.GenAI.USAGE_OUTPUT_TOKENS], 2)

    def test_trace_http_captures_basic_fields(self):
        class Response:
            status_code = 201

        @trace_http(name="http.call")
        def make_request(method: str, url: str):
            return Response()

        make_request("get", "https://example.com")

        span = self.exporter.get_finished_spans()[0]
        self.assertEqual(span.name, "http.call")
        self.assertEqual(span.attributes[semconv.HTTP.METHOD], "GET")
        self.assertEqual(span.attributes[semconv.HTTP.URL], "https://example.com")
        self.assertEqual(span.attributes[semconv.HTTP.STATUS_CODE], 201)
