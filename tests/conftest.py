"""Pytest fixtures for observability tests."""

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture
def setup_tracing():
    """
    Setup OpenTelemetry tracing for tests.

    This fixture can be used by tests that need real span recording,
    ensuring that spans are properly recorded and can be inspected.

    Usage:
        def test_something(setup_tracing):
            # test code here
    """
    # Save the current tracer provider to restore later
    original_provider = trace.get_tracer_provider()

    # Create an in-memory exporter
    exporter = InMemorySpanExporter()

    # Create a tracer provider with the exporter
    provider = TracerProvider()
    processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)

    # Set as the global tracer provider
    trace.set_tracer_provider(provider)

    yield exporter

    # Clean up after test
    exporter.clear()
    provider.shutdown()

    # Restore original tracer provider
    if original_provider and not isinstance(original_provider, trace.ProxyTracerProvider):
        trace.set_tracer_provider(original_provider)
