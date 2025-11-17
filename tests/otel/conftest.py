"""
Pytest fixtures for OpenTelemetry instrumentation tests.

Provides:
- In-memory span exporter for capturing OTEL spans
- TracerProvider setup for testing
- Environment-based control for real vs mock LLM calls
"""

import os

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

# Environment variable to enable real LLM API calls (disabled by default for CI)
RUN_REAL_LLM_TESTS = os.getenv("BASALT_RUN_REAL_LLM_TESTS") == "1"


@pytest.fixture
def otel_exporter():
    """
    In-memory span exporter for capturing OTEL spans during tests.

    This fixture provides an InMemorySpanExporter that will be attached to
    the TracerProvider created by InstrumentationManager. The exporter is
    automatically cleared before and after each test for isolation.

    Returns:
        InMemorySpanExporter: The exporter instance for span inspection.
    """
    exporter = InMemorySpanExporter()

    # Save the current tracer provider to restore later
    original_provider = trace.get_tracer_provider()

    yield exporter

    # Clean up after test
    exporter.clear()

    # Restore original tracer provider if it was real
    # Don't restore if it's just the default ProxyTracerProvider
    if original_provider and not isinstance(original_provider, trace.ProxyTracerProvider):
        trace.set_tracer_provider(original_provider)
