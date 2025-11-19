"""Tests for observability decorators."""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from unittest import mock

from basalt.observability import semconv
from basalt.observability.decorators import ObserveKind, observe
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


class ObserveDecoratorTestCase(unittest.TestCase):
    """Tests for the observe() universal decorator covering lines 592-642."""

    def setUp(self) -> None:
        self.exporter = get_exporter()
        self.exporter.clear()

    def _single_span(self):
        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        return spans[0]

    def test_observe_with_retrieval_kind_enum(self):
        """Test observe(kind=ObserveKind.RETRIEVAL) delegates to observe_retrieval."""
        @observe(kind=ObserveKind.RETRIEVAL, name="retrieval.search")
        def search(query: str) -> list[str]:
            return ["doc1", "doc2"]

        search("basalt framework")

        span = self._single_span()
        self.assertEqual(span.name, "retrieval.search")
        self.assertEqual(span.attributes[semconv.BasaltSpan.TYPE], "retrieval")
        self.assertEqual(span.attributes[semconv.BasaltRetrieval.QUERY], "basalt framework")

    def test_observe_with_retrieval_kind_string(self):
        """Test observe(kind='retrieval') delegates to observe_retrieval."""
        @observe(kind="retrieval", name="retrieval.search")
        def search(query: str) -> list[str]:
            return ["doc1", "doc2", "doc3"]

        search("python")

        span = self._single_span()
        self.assertEqual(span.name, "retrieval.search")
        self.assertEqual(span.attributes[semconv.BasaltSpan.TYPE], "retrieval")
        self.assertEqual(span.attributes[semconv.BasaltRetrieval.QUERY], "python")

    def test_observe_with_tool_kind_enum(self):
        """Test observe(kind=ObserveKind.TOOL) delegates to observe_tool."""
        @observe(kind=ObserveKind.TOOL, name="tool.invoke", tool_name=lambda bound: bound.arguments["name"])
        def invoke(name: str, payload: dict[str, str]) -> dict[str, bool]:
            return {"ok": True}

        invoke("calculator", {"operation": "add"})

        span = self._single_span()
        self.assertEqual(span.name, "tool.invoke")
        self.assertEqual(span.attributes[semconv.BasaltSpan.TYPE], "tool")
        self.assertEqual(span.attributes[semconv.BasaltTool.NAME], "calculator")
        tool_input = json.loads(span.attributes[semconv.BasaltTool.INPUT])
        self.assertEqual(tool_input["operation"], "add")

    def test_observe_with_tool_kind_string(self):
        """Test observe(kind='tool') delegates to observe_tool."""
        @observe(kind="tool", name="tool.execute", tool_name="search_engine")
        def execute(query: str) -> list[str]:
            return ["result1", "result2"]

        execute("test query")

        span = self._single_span()
        self.assertEqual(span.name, "tool.execute")
        self.assertEqual(span.attributes[semconv.BasaltSpan.TYPE], "tool")
        self.assertEqual(span.attributes[semconv.BasaltTool.NAME], "search_engine")

    def test_observe_with_event_kind_enum(self):
        """Test observe(kind=ObserveKind.EVENT) delegates to observe_event."""
        @observe(kind=ObserveKind.EVENT, name="event.notify", event_type="user-action")
        def notify(payload: dict[str, str]) -> dict[str, str]:
            return {"status": "sent"}

        notify({"message": "hello world"})

        span = self._single_span()
        self.assertEqual(span.name, "event.notify")
        self.assertEqual(span.attributes[semconv.BasaltSpan.TYPE], "event")
        self.assertEqual(span.attributes[semconv.BasaltEvent.TYPE], "user-action")
        event_input = json.loads(span.attributes[semconv.BasaltEvent.PAYLOAD])
        self.assertEqual(event_input["message"], "hello world")

    def test_observe_with_event_kind_string(self):
        """Test observe(kind='event') delegates to observe_event."""
        @observe(kind="event", name="event.log", event_type="system-event")
        def log_event(data: dict[str, int]) -> None:
            pass

        log_event({"count": 42})

        span = self._single_span()
        self.assertEqual(span.name, "event.log")
        self.assertEqual(span.attributes[semconv.BasaltSpan.TYPE], "event")
        self.assertEqual(span.attributes[semconv.BasaltEvent.TYPE], "system-event")

    def test_observe_with_function_kind_enum(self):
        """Test observe(kind=ObserveKind.FUNCTION) delegates to observe_function."""
        @observe(
            kind=ObserveKind.FUNCTION,
            name="compute.process",
            function_name=lambda bound: bound.arguments["task"],
            stage="preprocessing",
        )
        def process(task: str, values: list[int]) -> list[int]:
            return [val * 2 for val in values]

        process("multiplication", [1, 2, 3])

        span = self._single_span()
        self.assertEqual(span.name, "compute.process")
        self.assertEqual(span.attributes[semconv.BasaltSpan.TYPE], "function")
        self.assertEqual(span.attributes[semconv.BasaltFunction.NAME], "multiplication")
        self.assertEqual(span.attributes[semconv.BasaltFunction.STAGE], "preprocessing")

    def test_observe_with_function_kind_string(self):
        """Test observe(kind='function') delegates to observe_function."""
        @observe(kind="function", name="compute.transform", function_name="transformer", stage="postprocess")
        def transform(data: str) -> str:
            return data.upper()

        transform("hello")

        span = self._single_span()
        self.assertEqual(span.name, "compute.transform")
        self.assertEqual(span.attributes[semconv.BasaltSpan.TYPE], "function")
        self.assertEqual(span.attributes[semconv.BasaltFunction.NAME], "transformer")
        self.assertEqual(span.attributes[semconv.BasaltFunction.STAGE], "postprocess")

    def test_observe_with_generation_kind_enum(self):
        """Test observe(kind=ObserveKind.GENERATION) delegates to observe_generation."""
        @observe(kind=ObserveKind.GENERATION, name="llm.generate")
        def generate(model: str, prompt: str):
            return {
                "choices": [{"message": {"content": "response"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

        with mock.patch.dict(os.environ, {"TRACELOOP_TRACE_CONTENT": "true"}, clear=False):
            generate(model="gpt-4", prompt="What is AI?")

        span = self._single_span()
        self.assertEqual(span.name, "llm.generate")
        self.assertEqual(span.attributes[semconv.BasaltSpan.TYPE], "generation")
        self.assertEqual(span.attributes[semconv.GenAI.REQUEST_MODEL], "gpt-4")

    def test_observe_with_generation_kind_string(self):
        """Test observe(kind='generation') delegates to observe_generation."""
        @observe(kind="generation", name="llm.chat")
        def chat(model: str, messages: list[dict]):
            return {
                "choices": [{"message": {"content": "answer"}}],
                "usage": {"prompt_tokens": 15, "completion_tokens": 8},
            }

        with mock.patch.dict(os.environ, {"TRACELOOP_TRACE_CONTENT": "true"}, clear=False):
            chat(model="claude-3", messages=[{"role": "user", "content": "Hi"}])

        span = self._single_span()
        self.assertEqual(span.name, "llm.chat")
        self.assertEqual(span.attributes[semconv.BasaltSpan.TYPE], "generation")
        self.assertEqual(span.attributes[semconv.GenAI.REQUEST_MODEL], "claude-3")

    def test_observe_with_span_kind_enum(self):
        """Test observe(kind=ObserveKind.SPAN) delegates to observe_span."""
        @observe(
            kind=ObserveKind.SPAN,
            name="workflow.compute",
            variables=lambda bound: {"user": bound.arguments["user"]},
            evaluators=["quality"],
        )
        def compute(user: str, value: int) -> dict[str, int]:
            return {"total": value * 2}

        compute("alice", 5)

        span = self._single_span()
        self.assertEqual(span.name, "workflow.compute")
        self.assertEqual(span.attributes[semconv.BasaltSpan.TYPE], "span")

        variables_payload = json.loads(span.attributes[semconv.BasaltSpan.VARIABLES])
        self.assertEqual(variables_payload["user"], "alice")

        evaluators = span.attributes[semconv.BasaltSpan.EVALUATORS]
        self.assertIn("quality", evaluators)

    def test_observe_with_span_kind_string(self):
        """Test observe(kind='span') delegates to observe_span."""
        @observe(kind="span", name="workflow.process")
        def process(data: str) -> str:
            return data.lower()

        process("HELLO")

        span = self._single_span()
        self.assertEqual(span.name, "workflow.process")
        self.assertEqual(span.attributes[semconv.BasaltSpan.TYPE], "span")

    def test_observe_with_invalid_kind_raises_value_error(self):
        """Test observe() with invalid kind raises ValueError."""
        with self.assertRaises(ValueError) as context:
            @observe(kind="invalid_kind", name="test")
            def invalid_func():
                pass

        error_message = str(context.exception)
        self.assertIn("invalid_kind", error_message)
        self.assertIn("generation", error_message)
        self.assertIn("retrieval", error_message)
        self.assertIn("function", error_message)
        self.assertIn("tool", error_message)
        self.assertIn("event", error_message)
        self.assertIn("span", error_message)

    def test_observe_passes_common_parameters_retrieval(self):
        """Test observe() passes common parameters (name, attributes, variables) correctly."""
        @observe(
            kind=ObserveKind.RETRIEVAL,
            name="retrieval.advanced",
            attributes={"custom": "value"},
            variables=lambda bound: {"extracted": bound.arguments["query"]},
        )
        def search(query: str) -> list[str]:
            return ["result"]

        search("test query")

        span = self._single_span()
        self.assertEqual(span.name, "retrieval.advanced")
        self.assertEqual(span.attributes["custom"], "value")

        variables_payload = json.loads(span.attributes[semconv.BasaltSpan.VARIABLES])
        self.assertEqual(variables_payload["extracted"], "test query")

    def test_observe_async_function_retrieval(self):
        """Test observe() works with async functions for RETRIEVAL kind."""
        @observe(kind=ObserveKind.RETRIEVAL, name="async.retrieval")
        async def async_search(query: str) -> list[str]:
            await asyncio.sleep(0)
            return ["async_result"]

        asyncio.run(async_search("async query"))

        span = self._single_span()
        self.assertEqual(span.name, "async.retrieval")
        self.assertEqual(span.attributes[semconv.BasaltSpan.TYPE], "retrieval")

    def test_observe_async_function_tool(self):
        """Test observe() works with async functions for TOOL kind."""
        @observe(kind=ObserveKind.TOOL, name="async.tool", tool_name="async_calculator")
        async def async_tool(x: int, y: int) -> int:
            await asyncio.sleep(0)
            return x + y

        asyncio.run(async_tool(5, 3))

        span = self._single_span()
        self.assertEqual(span.name, "async.tool")
        self.assertEqual(span.attributes[semconv.BasaltSpan.TYPE], "tool")
        self.assertEqual(span.attributes[semconv.BasaltTool.NAME], "async_calculator")

    def test_observe_async_function_event(self):
        """Test observe() works with async functions for EVENT kind."""
        @observe(kind=ObserveKind.EVENT, name="async.event", event_type="async-action")
        async def async_event(data: dict) -> None:
            await asyncio.sleep(0)

        asyncio.run(async_event({"key": "value"}))

        span = self._single_span()
        self.assertEqual(span.name, "async.event")
        self.assertEqual(span.attributes[semconv.BasaltSpan.TYPE], "event")
        self.assertEqual(span.attributes[semconv.BasaltEvent.TYPE], "async-action")

    def test_observe_async_function_function(self):
        """Test observe() works with async functions for FUNCTION kind."""
        @observe(kind=ObserveKind.FUNCTION, name="async.function", function_name="async_processor", stage="async")
        async def async_function(value: int) -> int:
            await asyncio.sleep(0)
            return value * 2

        asyncio.run(async_function(10))

        span = self._single_span()
        self.assertEqual(span.name, "async.function")
        self.assertEqual(span.attributes[semconv.BasaltSpan.TYPE], "function")
        self.assertEqual(span.attributes[semconv.BasaltFunction.NAME], "async_processor")

    def test_observe_async_function_span(self):
        """Test observe() works with async functions for SPAN kind."""
        @observe(kind=ObserveKind.SPAN, name="async.span")
        async def async_span(value: str) -> str:
            await asyncio.sleep(0)
            return value.upper()

        asyncio.run(async_span("hello"))

        span = self._single_span()
        self.assertEqual(span.name, "async.span")
        self.assertEqual(span.attributes[semconv.BasaltSpan.TYPE], "span")

    def test_observe_context_manager_identity(self):
        """Identity parameter sets user and organization for context manager usage."""
        with observe(
            kind=ObserveKind.SPAN,
            name="identity.context",
            identity={
                "user": {"id": "user-ctx", "name": "Context User"},
                "organization": {"id": "org-ctx"},
            },
        ):
            pass

        span = self._single_span()
        self.assertEqual(span.attributes[semconv.BasaltUser.ID], "user-ctx")
        self.assertEqual(span.attributes[semconv.BasaltUser.NAME], "Context User")
        self.assertEqual(span.attributes[semconv.BasaltOrganization.ID], "org-ctx")

    def test_observe_decorator_identity_resolver(self):
        """Identity resolvers use bound arguments to populate user/org."""

        @observe(
            kind=ObserveKind.SPAN,
            name="identity.decorator",
            identity=lambda bound: {
                "user": {
                    "id": bound.arguments["user_id"],
                    "name": f"user-{bound.arguments['user_id']}",
                },
                "organization": bound.arguments["org_id"],
            },
        )
        def handler(user_id: str, org_id: str) -> str:
            return f"{user_id}:{org_id}"

        handler("alice", "acme")

        span = self._single_span()
        self.assertEqual(span.attributes[semconv.BasaltUser.ID], "alice")
        self.assertEqual(span.attributes[semconv.BasaltUser.NAME], "user-alice")
        self.assertEqual(span.attributes[semconv.BasaltOrganization.ID], "acme")
