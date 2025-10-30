"""Tests for simplified evaluator attachment (no registry/sampling)."""

from __future__ import annotations

import json
import unittest

from basalt.observability import (
    attach_evaluator,
    attach_evaluators_to_current_span,
    attach_evaluators_to_span,
    clear_trace_defaults,
    configure_trace_defaults,
    semconv,
    trace_generation,
    trace_span,
    update_current_span,
)
from tests.observability.utils import get_exporter


class EvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.exporter = get_exporter()
        self.exporter.clear()
        clear_trace_defaults()

    def tearDown(self) -> None:
        clear_trace_defaults()
        self.exporter.clear()

    def test_attach_evaluator_context_manager(self):
        """Test using attach_evaluator as context manager."""
        with trace_span("test.span") as span:
            with attach_evaluator("ctx-eval", span=span):
                pass

        span = self.exporter.get_finished_spans()[0]
        self.assertIn("ctx-eval", span.attributes[semconv.BasaltSpan.EVALUATORS])

    def test_attach_evaluator_to_current_span(self):
        """Test attaching evaluators to the current active span."""
        with trace_span("test.span"):
            attach_evaluators_to_current_span("current-eval")

        span = self.exporter.get_finished_spans()[0]
        self.assertIn("current-eval", span.attributes[semconv.BasaltSpan.EVALUATORS])

    def test_attach_evaluator_without_explicit_span(self):
        """Test attach_evaluator context manager finds current span automatically."""
        with trace_span("test.span"):
            with attach_evaluator("auto-eval"):
                pass

        span = self.exporter.get_finished_spans()[0]
        self.assertIn("auto-eval", span.attributes[semconv.BasaltSpan.EVALUATORS])

    def test_evaluators_from_trace_defaults(self):
        """Test that evaluators from trace defaults are applied with sampling."""
        configure_trace_defaults(evaluators=["default-eval"])

        with trace_span("test.span"):
            pass

        span = self.exporter.get_finished_spans()[0]
        self.assertIn("default-eval", span.attributes[semconv.BasaltSpan.EVALUATORS])
    # No per-evaluator attributes in simplified model

    def test_multiple_evaluators(self):
        """Test attaching multiple evaluators to a span."""
        with trace_span("test.span") as span:
            attach_evaluators_to_span(span, "eval-1", "eval-2", "eval-3")

        span = self.exporter.get_finished_spans()[0]
        evaluators = span.attributes[semconv.BasaltSpan.EVALUATORS]
        self.assertIn("eval-1", evaluators)
        self.assertIn("eval-2", evaluators)
        self.assertIn("eval-3", evaluators)
        # No per-evaluator attributes in simplified model

    def test_evaluator_uses_span_io(self):
        """Evaluator attachments rely on span-level IO only."""
        with trace_span("test.span") as span:
            span.set_input({"prompt": "hi"})
            span.set_output({"reply": "ok"})
            span.set_variables({"channel": "test"})
            attach_evaluators_to_span(span, "spec-eval")

        span = self.exporter.get_finished_spans()[0]
        self.assertIn("spec-eval", span.attributes[semconv.BasaltSpan.EVALUATORS])
        self.assertEqual(json.loads(span.attributes[semconv.BasaltSpan.INPUT])["prompt"], "hi")
        self.assertEqual(json.loads(span.attributes[semconv.BasaltSpan.OUTPUT])["reply"], "ok")
        self.assertEqual(json.loads(span.attributes[semconv.BasaltSpan.VARIABLES])["channel"], "test")

    def test_evaluator_with_llm_span(self):
        """Test attaching evaluators to LLM spans."""
        with trace_generation("test.llm") as span:
            span.set_model("gpt-4")
            attach_evaluators_to_span(span, "llm-eval")

        span = self.exporter.get_finished_spans()[0]
        evaluators = span.attributes[semconv.BasaltSpan.EVALUATORS]
        self.assertIn("llm-eval", evaluators)
        # No per-evaluator attributes in simplified model
        self.assertIn("llm-eval", span.attributes[semconv.BasaltSpan.EVALUATORS])
        self.assertEqual(span.attributes[semconv.GenAI.REQUEST_MODEL], "gpt-4")
        self.assertEqual(span.attributes[semconv.BasaltSpan.TYPE], "generation")

    def test_evaluator_with_llm_span_using_context_manager(self):
        """Test using attach_evaluator context manager with LLM spans."""
        with trace_generation("test.llm") as span:
            with attach_evaluator("llm-ctx-eval", span=span):
                span.set_model("gpt-4")
                span.set_prompt("Hello")
                span.set_completion("Hi there!")

        span = self.exporter.get_finished_spans()[0]
        self.assertIn("llm-ctx-eval", span.attributes[semconv.BasaltSpan.EVALUATORS])
        # No per-evaluator attributes in simplified model
        self.assertIn("llm-ctx-eval", span.attributes[semconv.BasaltSpan.EVALUATORS])
        self.assertEqual(span.attributes[semconv.GenAI.REQUEST_MODEL], "gpt-4")

    def test_unregistered_evaluator_defaults_to_100_percent(self):
        """Test that unregistered evaluators are attached with 100% probability."""
        with trace_span("test.span") as span:
            attach_evaluators_to_span(span, "unknown-eval")

        span = self.exporter.get_finished_spans()[0]
        self.assertIn("unknown-eval", span.attributes[semconv.BasaltSpan.EVALUATORS])
        # No per-evaluator attributes in simplified model

    # Sampling removed in simplified model

    def test_update_current_span_helper(self):
        """update_current_span populates IO and evaluator data."""

        with trace_span("test.span"):
            update_current_span(
                input_payload={"prompt": "hi"},
                output_payload={"reply": "ok"},
                variables={"channel": "helper"},
                evaluators=["helper-eval"],
            )

        span = self.exporter.get_finished_spans()[0]
        self.assertIn("helper-eval", span.attributes[semconv.BasaltSpan.EVALUATORS])
        self.assertEqual(json.loads(span.attributes[semconv.BasaltSpan.INPUT])["prompt"], "hi")
        self.assertEqual(json.loads(span.attributes[semconv.BasaltSpan.OUTPUT])["reply"], "ok")
        self.assertEqual(json.loads(span.attributes[semconv.BasaltSpan.VARIABLES])["channel"], "helper")
        # No per-evaluator attributes in simplified model

    def test_evaluator_propagation_to_child_spans(self):
        """Test that evaluators propagate to child spans via trace defaults."""
        configure_trace_defaults(evaluators=["parent-eval"])

        with trace_span("parent.span"):
            with trace_span("child.span"):
                pass

        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 2)

        # Both parent and child should have the evaluator
        for span in spans:
            self.assertIn("parent-eval", span.attributes[semconv.BasaltSpan.EVALUATORS])
            # No per-evaluator attributes in simplified model

    # Evaluator registry metadata removed in simplified model


if __name__ == "__main__":
    unittest.main()
