"""Tests for observability context managers."""

from __future__ import annotations

import json
import os
import unittest
from unittest import mock

from basalt.observability import (
    clear_trace_defaults,
    configure_trace_defaults,
    semconv,
    trace_event,
    trace_function,
    trace_generation,
    trace_retrieval,
    trace_span,
    trace_tool,
)
from tests.observability.utils import get_exporter


class ContextManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.exporter = get_exporter()
        self.exporter.clear()
        clear_trace_defaults()

    def tearDown(self) -> None:
        clear_trace_defaults()
        self.exporter.clear()

    def test_trace_span_records_custom_attributes(self):
        with trace_span("context.span", attributes={"component": "db"}) as span:
            span.set_attribute("rows", 5)
            span.add_event("cache_miss")

        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self.assertEqual(span.attributes["component"], "db")
        self.assertEqual(span.attributes["rows"], 5)

    def test_trace_generation_handles_helpers(self):
        with mock.patch.dict(os.environ, {"TRACELOOP_TRACE_CONTENT": "true"}, clear=False):
            with trace_generation("context.llm") as span:
                span.set_model("gpt-4")
                span.set_prompt("Hi")
                span.set_completion("Ok")
                span.set_tokens(input=10, output=1)

        span = self.exporter.get_finished_spans()[0]
        # Check new GenAI semantic conventions
        self.assertEqual(span.attributes[semconv.GenAI.REQUEST_MODEL], "gpt-4")

        # Check that prompt is stored as structured messages
        input_messages = json.loads(span.attributes[semconv.GenAI.INPUT_MESSAGES])
        self.assertEqual(len(input_messages), 1)
        self.assertEqual(input_messages[0]["role"], "user")
        self.assertEqual(input_messages[0]["parts"][0]["content"], "Hi")

        # Check that completion is stored as structured messages
        output_messages = json.loads(span.attributes[semconv.GenAI.OUTPUT_MESSAGES])
        self.assertEqual(len(output_messages), 1)
        self.assertEqual(output_messages[0]["role"], "assistant")
        self.assertEqual(output_messages[0]["parts"][0]["content"], "Ok")

        # Check token usage
        self.assertEqual(span.attributes[semconv.GenAI.USAGE_INPUT_TOKENS], 10)
        self.assertEqual(span.attributes[semconv.GenAI.USAGE_OUTPUT_TOKENS], 1)

        # Check span type
        self.assertEqual(span.attributes[semconv.BasaltSpan.TYPE], "generation")

    def test_trace_retrieval_helpers(self):
        with trace_retrieval("context.retrieval") as span:
            span.set_query("hello")
            span.set_results_count(3)
            span.set_top_k(5)

        span = self.exporter.get_finished_spans()[0]
        # Check new Basalt retrieval semantic conventions
        self.assertEqual(span.attributes[semconv.BasaltRetrieval.QUERY], "hello")
        self.assertEqual(span.attributes[semconv.BasaltRetrieval.RESULTS_COUNT], 3)
        self.assertEqual(span.attributes[semconv.BasaltRetrieval.TOP_K], 5)
        self.assertEqual(span.attributes[semconv.BasaltSpan.TYPE], "retrieval")

    def test_trace_tool_helpers(self):
        with trace_tool("context.tool") as span:
            span.set_tool_name("browser")
            span.set_input({"query": "hi"})
            span.set_output({"answer": "ok"})

        span = self.exporter.get_finished_spans()[0]
        self.assertEqual(span.attributes[semconv.BasaltTool.NAME], "browser")
        self.assertEqual(span.attributes[semconv.BasaltTool.INPUT], '{"query": "hi"}')
        self.assertEqual(span.attributes[semconv.BasaltTool.OUTPUT], '{"answer": "ok"}')
        self.assertEqual(span.attributes[semconv.BasaltSpan.TYPE], "tool")

    def test_trace_function_helpers(self):
        with trace_function("context.function") as span:
            span.set_function_name("score")
            span.set_stage("preprocess")
            span.add_metric("latency_ms", 42.5)

        span = self.exporter.get_finished_spans()[0]
        self.assertEqual(span.attributes[semconv.BasaltSpan.TYPE], "function")
        self.assertEqual(span.attributes[semconv.BasaltFunction.NAME], "score")
        self.assertEqual(span.attributes[semconv.BasaltFunction.STAGE], "preprocess")
        metric_key = f"{semconv.BasaltFunction.METRIC_PREFIX}.latency_ms"
        self.assertEqual(span.attributes[metric_key], 42.5)

    def test_trace_event_helpers(self):
        with trace_event("context.event", attributes={"source": "app"}) as span:
            span.set_event_type("workflow")
            span.set_payload({"status": "done"})

        span = self.exporter.get_finished_spans()[0]
        self.assertEqual(span.attributes["source"], "app")
        self.assertEqual(span.attributes[semconv.BasaltEvent.TYPE], "workflow")
        self.assertEqual(span.attributes[semconv.BasaltEvent.PAYLOAD], '{"status": "done"}')
        self.assertEqual(span.attributes[semconv.BasaltSpan.TYPE], "event")

    def test_trace_span_applies_default_context(self):
        configure_trace_defaults(
            user={"id": "user-1", "name": "Jane"},
            organization={"id": "org-1", "name": "Org"},
            experiment={"id": "exp-1", "feature_slug": "feature"},
            metadata={"env": "test"},
            evaluators=["eval-default"],
        )

        with trace_span("context.defaults") as span:
            span.add_evaluator("eval-inline")

        span = self.exporter.get_finished_spans()[0]
        self.assertEqual(span.attributes[semconv.BasaltUser.ID], "user-1")
        self.assertEqual(span.attributes[semconv.BasaltUser.NAME], "Jane")
        self.assertEqual(span.attributes[semconv.BasaltOrganization.ID], "org-1")
        self.assertEqual(span.attributes[semconv.BasaltExperiment.ID], "exp-1")
        self.assertEqual(span.attributes[semconv.BasaltExperiment.FEATURE_SLUG], "feature")
        self.assertEqual(span.attributes[f"{semconv.BASALT_META_PREFIX}env"], "test")
        evaluators = list(span.attributes[semconv.BasaltSpan.EVALUATORS])
        self.assertEqual(evaluators, ["eval-default", "eval-inline"])
        for slug in ("eval-default", "eval-inline"):
            self.assertEqual(
                span.attributes.get(f"{semconv.BasaltSpan.EVALUATOR_PREFIX}.{slug}.sample_rate"),
                1.0,
            )

    def test_variables_propagate_to_parent_span(self):
        with trace_span("parent.span"):
            with trace_span("child.span", variables={"prompt": "hello"}):
                pass

        spans = {span.name: span for span in self.exporter.get_finished_spans()}
        child_vars = json.loads(spans["child.span"].attributes[semconv.BasaltSpan.VARIABLES])
        parent_vars = json.loads(spans["parent.span"].attributes[semconv.BasaltSpan.VARIABLES])
        self.assertEqual(child_vars["prompt"], "hello")
        self.assertEqual(parent_vars["prompt"], "hello")
