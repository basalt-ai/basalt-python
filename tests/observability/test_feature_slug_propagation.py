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
    # Save the current tracer provider and directly manipulate the global state
    import opentelemetry.trace as trace_module
    
    original_provider = getattr(trace_module, '_TRACER_PROVIDER', None)
    original_set_once = getattr(trace_module, '_TRACER_PROVIDER_SET_ONCE', None)
    
    # Create a new Once object to allow setting a new provider
    from opentelemetry.util._once import Once
    trace_module._TRACER_PROVIDER_SET_ONCE = Once()
    trace_module._TRACER_PROVIDER = None

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    provider.add_span_processor(BasaltContextProcessor())
    provider.add_span_processor(BasaltCallEvaluatorProcessor())
    provider.add_span_processor(BasaltShouldEvaluateProcessor())
    provider.add_span_processor(BasaltAutoInstrumentationProcessor())

    trace.set_tracer_provider(provider)

    yield exporter

    # Clean up
    exporter.clear()
    provider.shutdown()
    
    # Restore original state
    trace_module._TRACER_PROVIDER_SET_ONCE = Once()
    trace_module._TRACER_PROVIDER = None
    
    if original_provider and not isinstance(original_provider, trace.ProxyTracerProvider):
        trace.set_tracer_provider(original_provider)
    
    # Restore the original set_once object if it existed
    if original_set_once is not None:
        trace_module._TRACER_PROVIDER_SET_ONCE = original_set_once


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


def test_feature_slug_propagates_to_api_request_spans(setup_feature_slug_tracer):
    """Ensure feature_slug from start_observe reaches API request spans created via observe()."""
    exporter = setup_feature_slug_tracer

    with start_observe(name="root", feature_slug="test-feature"):
        # Simulate an API request span using observe (as used in request_tracing.py)
        with observe(name="api.prompts.get", kind=ObserveKind.SPAN):
            pass

        # Simulate nested observe calls
        with observe(name="nested.operation", kind=ObserveKind.FUNCTION):
            with observe(name="deeply.nested", kind=ObserveKind.SPAN):
                pass

    spans = exporter.get_finished_spans()
    span_by_name = {span.name: span for span in spans}

    expected_names = {
        "root",
        "api.prompts.get",
        "nested.operation",
        "deeply.nested",
    }
    assert expected_names.issubset(span_by_name.keys()), (
        f"Missing spans: {expected_names - set(span_by_name.keys())}"
    )

    # Verify all spans have the feature_slug
    for name in expected_names:
        span = span_by_name[name]
        assert span.attributes.get(BasaltSpan.FEATURE_SLUG) == "test-feature", (
            f"Span '{name}' missing feature_slug propagation. "
            f"Expected 'test-feature', got {span.attributes.get(BasaltSpan.FEATURE_SLUG)}"
        )


def test_feature_slug_propagates_to_decorated_functions(setup_feature_slug_tracer):
    """Ensure feature_slug from start_observe reaches decorated function spans."""
    exporter = setup_feature_slug_tracer

    @observe(name="decorated_function", kind=ObserveKind.FUNCTION)
    def my_function():
        return "result"

    @observe(name="decorated_async_function", kind=ObserveKind.FUNCTION)
    async def my_async_function():
        return "async_result"

    with start_observe(name="root", feature_slug="decorator-test"):
        my_function()

    spans = exporter.get_finished_spans()
    span_by_name = {span.name: span for span in spans}

    # Verify decorated function span has feature_slug
    assert "decorated_function" in span_by_name, "decorated_function span not found"
    function_span = span_by_name["decorated_function"]
    assert function_span.attributes.get(BasaltSpan.FEATURE_SLUG) == "decorator-test", (
        f"Decorated function span missing feature_slug propagation. "
        f"Expected 'decorator-test', got {function_span.attributes.get(BasaltSpan.FEATURE_SLUG)}"
    )

    # Clear for next test
    exporter.clear()

    # Test async decorated function
    import asyncio

    async def test_async():
        with start_observe(name="async_root", feature_slug="async-decorator-test"):
            await my_async_function()

    asyncio.run(test_async())

    spans = exporter.get_finished_spans()
    span_by_name = {span.name: span for span in spans}

    # Verify async decorated function span has feature_slug
    assert "decorated_async_function" in span_by_name, "decorated_async_function span not found"
    async_function_span = span_by_name["decorated_async_function"]
    assert async_function_span.attributes.get(BasaltSpan.FEATURE_SLUG) == "async-decorator-test", (
        f"Async decorated function span missing feature_slug propagation. "
        f"Expected 'async-decorator-test', got {async_function_span.attributes.get(BasaltSpan.FEATURE_SLUG)}"
    )
