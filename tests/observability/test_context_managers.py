"""Tests for observability context managers."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from basalt.observability.context_managers import trace_llm_call, trace_retrieval, trace_span
from tests.observability.utils import get_exporter


class ContextManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.exporter = get_exporter()
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

    def test_trace_llm_call_handles_helpers(self):
        with mock.patch.dict(os.environ, {"TRACELOOP_TRACE_CONTENT": "true"}, clear=False):
            with trace_llm_call("context.llm") as span:
                span.set_model("gpt-4")
                span.set_prompt("Hi")
                span.set_completion("Ok")
                span.set_tokens(input=10, output=1)

        span = self.exporter.get_finished_spans()[0]
        self.assertEqual(span.attributes["llm.model"], "gpt-4")
        self.assertEqual(span.attributes["llm.prompt"], "Hi")
        self.assertEqual(span.attributes["llm.completion"], "Ok")
        self.assertEqual(span.attributes["llm.tokens.input"], 10)
        self.assertEqual(span.attributes["llm.tokens.output"], 1)

    def test_trace_retrieval_helpers(self):
        with trace_retrieval("context.retrieval") as span:
            span.set_query("hello")
            span.set_results_count(3)
            span.set_top_k(5)

        span = self.exporter.get_finished_spans()[0]
        self.assertEqual(span.attributes["retrieval.query"], "hello")
        self.assertEqual(span.attributes["retrieval.results.count"], 3)
        self.assertEqual(span.attributes["retrieval.top_k"], 5)
