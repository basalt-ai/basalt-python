"""Tests for Basalt OpenTelemetry span processors."""

from __future__ import annotations

import unittest
from collections.abc import Sequence
from typing import cast

from opentelemetry import trace

from basalt.observability import (
    clear_trace_defaults,
    configure_trace_defaults,
    evaluator,
    semconv,
    with_evaluators,
)
from tests.observability.utils import get_exporter


class ProcessorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.exporter = get_exporter()
        self.exporter.clear()
        clear_trace_defaults()

    def tearDown(self) -> None:
        clear_trace_defaults()
        self.exporter.clear()

    def test_defaults_apply_to_auto_instrumented_spans(self) -> None:
        configure_trace_defaults(
            user={"id": "user-42"},
            organization={"id": "org-9"},
            metadata={"region": "us-east"},
            evaluators=["default-processor"],
        )

        tracer = trace.get_tracer("tests.processors")
        with tracer.start_as_current_span("auto.span"):
            pass

        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        attrs = span.attributes or {}
        self.assertEqual(attrs[semconv.BasaltUser.ID], "user-42")
        self.assertEqual(attrs[semconv.BasaltOrganization.ID], "org-9")
        self.assertEqual(attrs[f"{semconv.BASALT_META_PREFIX}region"], "us-east")
        raw_evaluators = attrs[semconv.BasaltSpan.EVALUATORS]
        self.assertIsInstance(raw_evaluators, (list, tuple))
        evaluators = list(cast(Sequence[str], raw_evaluators))
        self.assertIn("default-processor", evaluators)

    def test_with_evaluators_propagates_to_auto_spans(self) -> None:
        tracer = trace.get_tracer("tests.processors")
        with with_evaluators(["ctx-eval"]):
            with tracer.start_as_current_span("auto.ctx.span"):
                pass

        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        attrs = span.attributes or {}
        raw_evaluators = attrs[semconv.BasaltSpan.EVALUATORS]
        self.assertIsInstance(raw_evaluators, (list, tuple))
        evaluators = list(cast(Sequence[str], raw_evaluators))
        self.assertEqual(evaluators, ["ctx-eval"])

    def test_evaluator_decorator_propagates_to_inner_spans(self) -> None:
        tracer = trace.get_tracer("tests.processors")

        @evaluator("decorator-eval")
        def invoke() -> None:
            with tracer.start_as_current_span("decorated.auto.span"):
                pass

        invoke()

        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        attrs = spans[0].attributes or {}
        raw_evaluators = attrs[semconv.BasaltSpan.EVALUATORS]
        self.assertIsInstance(raw_evaluators, (list, tuple))
        evaluators = list(cast(Sequence[str], raw_evaluators))
        self.assertEqual(evaluators, ["decorator-eval"])


if __name__ == "__main__":
    unittest.main()
