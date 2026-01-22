"""Tests for multiple span exporters functionality."""

from __future__ import annotations

from unittest import mock

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from basalt.observability.config import TelemetryConfig
from basalt.observability.instrumentation import (
    BasaltConfig,
    InstrumentationManager,
    create_tracer_provider,
)


@pytest.fixture(scope="module", autouse=True)
def restore_trace_provider():
    """Restore the original global tracer provider after this module."""
    original_provider = trace.get_tracer_provider()
    original_once = trace._TRACER_PROVIDER_SET_ONCE

    yield

    trace._TRACER_PROVIDER_SET_ONCE = original_once
    if original_provider and not isinstance(original_provider, trace.ProxyTracerProvider):
        trace._TRACER_PROVIDER = None
        trace._TRACER_PROVIDER_SET_ONCE = trace.Once()
        try:
            trace.set_tracer_provider(original_provider)
        except Exception:
            pass


@pytest.fixture(autouse=True)
def reset_trace_provider():
    """Reset provider state before each test so we can set new ones."""
    trace._TRACER_PROVIDER = None
    trace._TRACER_PROVIDER_SET_ONCE = trace.Once()
    yield


def test_single_exporter_backward_compatibility():
    """Test that single exporter still works (backward compatibility)."""
    exporter = InMemorySpanExporter()
    config = BasaltConfig(service_name="test-service")

    provider = create_tracer_provider(config, exporter=exporter)

    # Verify provider was created
    assert isinstance(provider, TracerProvider)
    # Verify exporter was added (check _active_span_processor has processors)
    assert len(provider._active_span_processor._span_processors) > 0


def test_multiple_exporters_list():
    """Test configuring with list of 2 exporters."""
    exporter1 = InMemorySpanExporter()
    exporter2 = InMemorySpanExporter()
    config = BasaltConfig(service_name="test-service")

    provider = create_tracer_provider(config, exporter=[exporter1, exporter2])

    # Verify provider was created
    assert isinstance(provider, TracerProvider)
    # Verify both exporters were added (2 processors)
    assert len(provider._active_span_processor._span_processors) == 2

    # Test that both exporters receive spans
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer("test")

    with tracer.start_as_current_span("test-span"):
        pass

    # Force flush to ensure spans are exported
    provider.force_flush()

    # Both exporters should have received the span
    assert len(exporter1.get_finished_spans()) == 1
    assert len(exporter2.get_finished_spans()) == 1

    # Verify span content is identical
    span1 = exporter1.get_finished_spans()[0]
    span2 = exporter2.get_finished_spans()[0]
    assert span1.name == span2.name
    assert span1.context.trace_id == span2.context.trace_id
    assert span1.context.span_id == span2.context.span_id


def test_empty_list_uses_console_exporter():
    """Test that empty list falls back to ConsoleSpanExporter with warning."""
    config = BasaltConfig(service_name="test-service")

    with pytest.warns(UserWarning) as warning:
        provider = create_tracer_provider(config, exporter=[])

    # Verify warning message
    assert "Empty exporter list" in str(warning[0].message)

    # Verify ConsoleSpanExporter was used
    assert isinstance(provider, TracerProvider)
    # Check that a processor was added
    assert len(provider._active_span_processor._span_processors) > 0


@mock.patch.dict(
    "os.environ", {"BASALT_OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4318"}, clear=False
)
@mock.patch("basalt.observability.instrumentation.OTLPSpanExporter")
def test_user_exporters_plus_env_exporter(mock_otlp_exporter):
    """Test that user exporters are used instead of environment exporter."""
    # Mock the OTLP exporter creation
    mock_env_exporter = mock.Mock()
    mock_otlp_exporter.return_value = mock_env_exporter

    user_exporter = InMemorySpanExporter()
    config = TelemetryConfig(
        service_name="test-service",
        exporter=user_exporter,
    )

    manager = InstrumentationManager()
    manager.initialize(config)

    # Verify user exporter was used
    provider = manager._tracer_provider
    assert isinstance(provider, TracerProvider)
    # Should have 1 exporter + 4 Basalt processors = 5 total processors
    # Basalt processors: Context, CallEvaluator, ShouldEvaluate, AutoInstrumentation
    # Note: Environment exporter is only used if no user exporter is provided
    assert len(provider._active_span_processor._span_processors) == 5


def test_mixed_console_and_otlp_exporters():
    """Test mix of ConsoleSpanExporter and regular exporters."""
    console_exporter = ConsoleSpanExporter()
    memory_exporter = InMemorySpanExporter()
    config = BasaltConfig(service_name="test-service")

    provider = create_tracer_provider(config, exporter=[console_exporter, memory_exporter])

    # Verify both exporters were added
    assert isinstance(provider, TracerProvider)
    assert len(provider._active_span_processor._span_processors) == 2

    # Test span export
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer("test")

    with tracer.start_as_current_span("test-span"):
        pass

    provider.force_flush()

    # Memory exporter should have received the span
    assert len(memory_exporter.get_finished_spans()) == 1


def test_exporter_isolation_on_error():
    """Test that one failing exporter doesn't affect others."""
    failing_exporter = mock.Mock()
    failing_exporter.export.side_effect = Exception("Export failed")

    working_exporter = InMemorySpanExporter()
    config = BasaltConfig(service_name="test-service")

    provider = create_tracer_provider(config, exporter=[failing_exporter, working_exporter])

    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer("test")

    with tracer.start_as_current_span("test-span"):
        pass

    # Force flush (failing exporter will raise but shouldn't stop working exporter)
    try:
        provider.force_flush()
    except Exception:
        pass  # Expected from failing exporter

    # Working exporter should still have received the span
    assert len(working_exporter.get_finished_spans()) == 1


def test_duplicate_exporters_allowed():
    """Test that duplicate exporters in list are allowed (user responsibility)."""
    exporter = InMemorySpanExporter()
    config = BasaltConfig(service_name="test-service")

    # Same exporter instance twice
    provider = create_tracer_provider(config, exporter=[exporter, exporter])

    # Should have 2 processors (both using same exporter)
    assert len(provider._active_span_processor._span_processors) == 2


def test_none_exporter_uses_console_with_warning():
    """Test that None exporter defaults to ConsoleSpanExporter with warning."""
    config = BasaltConfig(service_name="test-service")

    with pytest.warns(UserWarning) as warning:
        provider = create_tracer_provider(config, exporter=None)

    # Verify warning message
    assert "No span exporter configured" in str(warning[0].message)

    # Verify provider was created
    assert isinstance(provider, TracerProvider)


def test_config_accepts_exporter_list():
    """Test that TelemetryConfig accepts list of exporters."""
    exporter1 = InMemorySpanExporter()
    exporter2 = InMemorySpanExporter()

    config = TelemetryConfig(
        service_name="test-service",
        exporter=[exporter1, exporter2],
    )

    assert isinstance(config.exporter, list)
    assert len(config.exporter) == 2
    assert config.exporter[0] is exporter1
    assert config.exporter[1] is exporter2


def test_config_accepts_single_exporter():
    """Test backward compatibility: single exporter still works."""
    exporter = InMemorySpanExporter()

    config = TelemetryConfig(
        service_name="test-service",
        exporter=exporter,
    )

    # Should be the exporter itself, not wrapped in list
    assert isinstance(config.exporter, InMemorySpanExporter)
    assert config.exporter is exporter


def test_clone_with_exporter_list():
    """Test that clone() properly copies exporter lists."""
    exporter1 = InMemorySpanExporter()
    exporter2 = InMemorySpanExporter()

    original = TelemetryConfig(
        service_name="test-service",
        exporter=[exporter1, exporter2],
    )

    cloned = original.clone()

    # Verify it's a new list instance
    assert cloned.exporter is not original.exporter
    # But contains same exporter objects
    assert len(cloned.exporter) == 2
    assert cloned.exporter[0] is exporter1
    assert cloned.exporter[1] is exporter2


def test_clone_list_independence():
    """Test that modifying cloned exporter list doesn't affect original."""
    exporter1 = InMemorySpanExporter()
    exporter2 = InMemorySpanExporter()

    original = TelemetryConfig(
        service_name="test-service",
        exporter=[exporter1, exporter2],
    )

    cloned = original.clone()

    # Modify cloned list
    if isinstance(cloned.exporter, list):
        cloned.exporter.append(InMemorySpanExporter())

    # Original should be unchanged
    assert len(original.exporter) == 2


def test_clone_with_single_exporter():
    """Test that clone() handles single exporter correctly."""
    exporter = InMemorySpanExporter()

    original = TelemetryConfig(
        service_name="test-service",
        exporter=exporter,
    )

    cloned = original.clone()

    # Should be same exporter object (not cloned)
    assert cloned.exporter is exporter
