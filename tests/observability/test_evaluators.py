"""Tests for evaluator sampling and attachment."""

from __future__ import annotations

import unittest

from basalt.observability import (
    EvaluatorConfig,
    attach_evaluator,
    attach_evaluators_to_current_span,
    attach_evaluators_to_span,
    clear_trace_defaults,
    configure_trace_defaults,
    get_evaluator_manager,
    register_evaluator,
    semconv,
    trace_llm_call,
    trace_span,
    unregister_evaluator,
)
from tests.observability.utils import get_exporter


class EvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.exporter = get_exporter()
        self.exporter.clear()
        clear_trace_defaults()
        # Clear all registered evaluators
        manager = get_evaluator_manager()
        for slug in list(manager.list_evaluators()):
            unregister_evaluator(slug)

    def tearDown(self) -> None:
        clear_trace_defaults()
        self.exporter.clear()
        # Clean up registered evaluators
        manager = get_evaluator_manager()
        for slug in list(manager.list_evaluators()):
            unregister_evaluator(slug)

    def test_evaluator_config_validation(self):
        """Test that EvaluatorConfig validates inputs."""
        # Valid config
        config = EvaluatorConfig(slug="test-eval", sample_rate=0.5)
        self.assertEqual(config.slug, "test-eval")
        self.assertEqual(config.sample_rate, 0.5)

        # Invalid sample rate
        with self.assertRaises(ValueError):
            EvaluatorConfig(slug="test", sample_rate=1.5)

        with self.assertRaises(ValueError):
            EvaluatorConfig(slug="test", sample_rate=-0.1)

        # Empty slug
        with self.assertRaises(ValueError):
            EvaluatorConfig(slug="", sample_rate=1.0)

    def test_register_and_unregister_evaluator(self):
        """Test registering and unregistering evaluators."""
        register_evaluator("test-eval", sample_rate=0.5)
        manager = get_evaluator_manager()

        config = manager.get_config("test-eval")
        self.assertIsNotNone(config)
        self.assertEqual(config.slug, "test-eval")
        self.assertEqual(config.sample_rate, 0.5)

        # Unregister
        unregister_evaluator("test-eval")
        config = manager.get_config("test-eval")
        self.assertIsNone(config)

    def test_evaluator_with_100_percent_sample_rate(self):
        """Test that evaluators with 100% sample rate are always attached."""
        register_evaluator("always-eval", sample_rate=1.0)

        with trace_span("test.span") as span:
            attach_evaluators_to_span(span, "always-eval")

        span = self.exporter.get_finished_spans()[0]
        self.assertIn("always-eval", span.attributes["basalt.trace.evaluators"])

    def test_evaluator_with_0_percent_sample_rate(self):
        """Test that evaluators with 0% sample rate are never attached."""
        register_evaluator("never-eval", sample_rate=0.0)

        with trace_span("test.span") as span:
            attach_evaluators_to_span(span, "never-eval")

        span = self.exporter.get_finished_spans()[0]
        # Should not have evaluators attribute or should not contain the evaluator
        evaluators = span.attributes.get("basalt.trace.evaluators", [])
        self.assertNotIn("never-eval", evaluators)

    def test_attach_evaluator_context_manager(self):
        """Test using attach_evaluator as context manager."""
        register_evaluator("ctx-eval", sample_rate=1.0)

        with trace_span("test.span") as span:
            with attach_evaluator("ctx-eval", span=span):
                pass

        span = self.exporter.get_finished_spans()[0]
        self.assertIn("ctx-eval", span.attributes["basalt.trace.evaluators"])

    def test_attach_evaluator_to_current_span(self):
        """Test attaching evaluators to the current active span."""
        register_evaluator("current-eval", sample_rate=1.0)

        with trace_span("test.span"):
            attach_evaluators_to_current_span("current-eval")

        span = self.exporter.get_finished_spans()[0]
        self.assertIn("current-eval", span.attributes["basalt.trace.evaluators"])

    def test_attach_evaluator_without_explicit_span(self):
        """Test attach_evaluator context manager finds current span automatically."""
        register_evaluator("auto-eval", sample_rate=1.0)

        with trace_span("test.span"):
            with attach_evaluator("auto-eval"):
                pass

        span = self.exporter.get_finished_spans()[0]
        self.assertIn("auto-eval", span.attributes["basalt.trace.evaluators"])

    def test_evaluators_from_trace_defaults(self):
        """Test that evaluators from trace defaults are applied with sampling."""
        register_evaluator("default-eval", sample_rate=1.0)

        configure_trace_defaults(evaluators=["default-eval"])

        with trace_span("test.span"):
            pass

        span = self.exporter.get_finished_spans()[0]
        self.assertIn("default-eval", span.attributes["basalt.trace.evaluators"])

    def test_multiple_evaluators(self):
        """Test attaching multiple evaluators to a span."""
        register_evaluator("eval-1", sample_rate=1.0)
        register_evaluator("eval-2", sample_rate=1.0)
        register_evaluator("eval-3", sample_rate=1.0)

        with trace_span("test.span") as span:
            attach_evaluators_to_span(span, "eval-1", "eval-2", "eval-3")

        span = self.exporter.get_finished_spans()[0]
        evaluators = span.attributes["basalt.trace.evaluators"]
        self.assertIn("eval-1", evaluators)
        self.assertIn("eval-2", evaluators)
        self.assertIn("eval-3", evaluators)

    def test_evaluator_with_llm_span(self):
        """Test attaching evaluators to LLM spans."""
        register_evaluator("llm-eval", sample_rate=1.0)

        with trace_llm_call("test.llm") as span:
            span.set_model("gpt-4")
            attach_evaluators_to_span(span, "llm-eval")

        span = self.exporter.get_finished_spans()[0]
        self.assertIn("llm-eval", span.attributes[semconv.BasaltTrace.EVALUATORS])
        self.assertEqual(span.attributes[semconv.GenAI.REQUEST_MODEL], "gpt-4")
        self.assertEqual(span.attributes[semconv.BasaltSpan.TYPE], "generation")

    def test_evaluator_with_llm_span_using_context_manager(self):
        """Test using attach_evaluator context manager with LLM spans."""
        register_evaluator("llm-ctx-eval", sample_rate=1.0)

        with trace_llm_call("test.llm") as span:
            with attach_evaluator("llm-ctx-eval", span=span):
                span.set_model("gpt-4")
                span.set_prompt("Hello")
                span.set_completion("Hi there!")

        span = self.exporter.get_finished_spans()[0]
        self.assertIn("llm-ctx-eval", span.attributes[semconv.BasaltTrace.EVALUATORS])
        self.assertEqual(span.attributes[semconv.GenAI.REQUEST_MODEL], "gpt-4")

    def test_unregistered_evaluator_defaults_to_100_percent(self):
        """Test that unregistered evaluators are attached with 100% probability."""
        # Don't register "unknown-eval"

        with trace_span("test.span") as span:
            attach_evaluators_to_span(span, "unknown-eval")

        span = self.exporter.get_finished_spans()[0]
        # Should be attached since unregistered evaluators default to 100%
        self.assertIn("unknown-eval", span.attributes["basalt.trace.evaluators"])

    def test_evaluator_sampling_statistical(self):
        """Test that sampling works statistically (not deterministic)."""
        register_evaluator("half-eval", sample_rate=0.5)

        attached_count = 0
        total_runs = 100

        for _ in range(total_runs):
            self.exporter.clear()
            with trace_span("test.span") as span:
                attach_evaluators_to_span(span, "half-eval")

            span = self.exporter.get_finished_spans()[0]
            evaluators = span.attributes.get("basalt.trace.evaluators", [])
            if "half-eval" in evaluators:
                attached_count += 1

        # With 50% sample rate, we expect roughly 50 out of 100
        # Allow for statistical variance (e.g., 30-70 range)
        self.assertGreater(attached_count, 20)
        self.assertLess(attached_count, 80)

    def test_evaluator_propagation_to_child_spans(self):
        """Test that evaluators propagate to child spans via trace defaults."""
        register_evaluator("parent-eval", sample_rate=1.0)
        configure_trace_defaults(evaluators=["parent-eval"])

        with trace_span("parent.span"):
            with trace_span("child.span"):
                pass

        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 2)

        # Both parent and child should have the evaluator
        for span in spans:
            self.assertIn("parent-eval", span.attributes["basalt.trace.evaluators"])

    def test_evaluator_metadata(self):
        """Test that evaluator metadata can be stored."""
        register_evaluator(
            "meta-eval",
            sample_rate=1.0,
            metadata={"version": "1.0", "type": "quality"},
        )

        manager = get_evaluator_manager()
        config = manager.get_config("meta-eval")

        self.assertIsNotNone(config)
        self.assertEqual(config.metadata["version"], "1.0")
        self.assertEqual(config.metadata["type"], "quality")


if __name__ == "__main__":
    unittest.main()
