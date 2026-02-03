"""Tests for feature_slug propagation across Basalt and auto-instrumented spans."""

from __future__ import annotations

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from basalt.observability import observe, start_observe
from basalt.observability.decorators import ObserveKind
from basalt.observability.processors import (
    BasaltAutoInstrumentationProcessor,
    BasaltCallEvaluatorProcessor,
    BasaltContextProcessor,
    BasaltShouldEvaluateProcessor,
)
from basalt.observability.semconv import BasaltSpan


@pytest.fixture(scope="function")
def setup_feature_slug_tracer():
    """Install Basalt processors + in-memory exporter for feature_slug propagation tests."""
    original_provider = trace.get_tracer_provider()

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    provider.add_span_processor(BasaltContextProcessor())
    provider.add_span_processor(BasaltCallEvaluatorProcessor())
    provider.add_span_processor(BasaltShouldEvaluateProcessor())
    provider.add_span_processor(BasaltAutoInstrumentationProcessor())

    trace.set_tracer_provider(provider)

    yield exporter

    exporter.clear()
    provider.shutdown()

    if original_provider and not isinstance(original_provider, trace.ProxyTracerProvider):
        trace.set_tracer_provider(original_provider)


def test_feature_slug_propagates_to_basalt_fastapi_and_autoinstrumented_spans(
    setup_feature_slug_tracer,
):
    """Ensure feature_slug from start_observe reaches Basalt + surrounding spans."""
    exporter = setup_feature_slug_tracer

    tracer_fastapi = trace.get_tracer("opentelemetry.instrumentation.fastapi")
    tracer_openai = trace.get_tracer("opentelemetry.instrumentation.openai.v1")
    tracer_httpx = trace.get_tracer("opentelemetry.instrumentation.httpx")

    with tracer_fastapi.start_as_current_span("fastapi.request"):
        with start_observe(name="handler", feature_slug="support-ticket"):
            with observe(name="child", kind=ObserveKind.FUNCTION):
                pass

            with tracer_openai.start_as_current_span("openai.call"):
                pass

            with tracer_httpx.start_as_current_span("httpx.request"):
                pass

    spans = exporter.get_finished_spans()
    span_by_name = {span.name: span for span in spans}

    expected_names = {
        "fastapi.request",
        "handler",
        "child",
        "openai.call",
        "httpx.request",
    }
    assert expected_names.issubset(span_by_name.keys()), (
        f"Missing spans: {expected_names - set(span_by_name.keys())}"
    )

    for name in expected_names:
        span = span_by_name[name]
        assert span.attributes.get(BasaltSpan.FEATURE_SLUG) == "support-ticket", (
            f"Span {name} missing feature_slug propagation"
        )
