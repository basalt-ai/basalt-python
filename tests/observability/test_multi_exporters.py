"""Tests for multiple span exporters functionality."""

import unittest
from unittest import mock

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


class TestMultipleExporters(unittest.TestCase):
    """Test cases for multiple span exporters support."""

    @classmethod
    def setUpClass(cls):
        """Save original global state before any tests run."""
        cls._original_provider = trace.get_tracer_provider()
        cls._original_once = trace._TRACER_PROVIDER_SET_ONCE

    @classmethod
    def tearDownClass(cls):
        """Restore original global state after all tests complete."""
        # Restore the Once flag so other test files can set providers
        trace._TRACER_PROVIDER_SET_ONCE = cls._original_once
        # If there was a real provider originally, try to restore it
        if cls._original_provider and not isinstance(cls._original_provider, trace.ProxyTracerProvider):
            trace._TRACER_PROVIDER = None
            trace._TRACER_PROVIDER_SET_ONCE = trace.Once()
            try:
                trace.set_tracer_provider(cls._original_provider)
            except Exception:
                pass

    def setUp(self):
        """Reset provider state before each test so we can set new ones."""
        # Allow each test to set its own provider by resetting the Once flag
        trace._TRACER_PROVIDER = None
        trace._TRACER_PROVIDER_SET_ONCE = trace.Once()

    def tearDown(self):
        """Don't clean up - let setUpClass/tearDownClass handle global state."""
        pass

    def test_single_exporter_backward_compatibility(self):
        """Test that single exporter still works (backward compatibility)."""
        exporter = InMemorySpanExporter()
        config = BasaltConfig(service_name="test-service")

        provider = create_tracer_provider(config, exporter=exporter)

        # Verify provider was created
        self.assertIsInstance(provider, TracerProvider)
        # Verify exporter was added (check _active_span_processor has processors)
        self.assertGreater(len(provider._active_span_processor._span_processors), 0)

    def test_multiple_exporters_list(self):
        """Test configuring with list of 2 exporters."""
        exporter1 = InMemorySpanExporter()
        exporter2 = InMemorySpanExporter()
        config = BasaltConfig(service_name="test-service")

        provider = create_tracer_provider(config, exporter=[exporter1, exporter2])

        # Verify provider was created
        self.assertIsInstance(provider, TracerProvider)
        # Verify both exporters were added (2 processors)
        self.assertEqual(len(provider._active_span_processor._span_processors), 2)

        # Test that both exporters receive spans
        trace.set_tracer_provider(provider)
        tracer = trace.get_tracer("test")

        with tracer.start_as_current_span("test-span"):
            pass

        # Force flush to ensure spans are exported
        provider.force_flush()

        # Both exporters should have received the span
        self.assertEqual(len(exporter1.get_finished_spans()), 1)
        self.assertEqual(len(exporter2.get_finished_spans()), 1)

        # Verify span content is identical
        span1 = exporter1.get_finished_spans()[0]
        span2 = exporter2.get_finished_spans()[0]
        self.assertEqual(span1.name, span2.name)
        self.assertEqual(span1.context.trace_id, span2.context.trace_id)
        self.assertEqual(span1.context.span_id, span2.context.span_id)

    def test_empty_list_uses_console_exporter(self):
        """Test that empty list falls back to ConsoleSpanExporter with warning."""
        config = BasaltConfig(service_name="test-service")

        with self.assertWarns(UserWarning) as cm:
            provider = create_tracer_provider(config, exporter=[])

        # Verify warning message
        self.assertIn("Empty exporter list", str(cm.warning))

        # Verify ConsoleSpanExporter was used
        self.assertIsInstance(provider, TracerProvider)
        # Check that a processor was added
        self.assertGreater(len(provider._active_span_processor._span_processors), 0)

    @mock.patch.dict("os.environ", {"BASALT_OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4318"}, clear=False)
    @mock.patch("basalt.observability.instrumentation.OTLPSpanExporter")
    def test_user_exporters_plus_env_exporter(self, mock_otlp_exporter):
        """Test that user exporters are combined with environment exporter."""
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

        # Verify both exporters were used
        provider = manager._tracer_provider
        self.assertIsInstance(provider, TracerProvider)
        # Should have 2 exporters + 3 Basalt processors = 5 total processors
        self.assertEqual(len(provider._active_span_processor._span_processors), 5)

    def test_mixed_console_and_otlp_exporters(self):
        """Test mix of ConsoleSpanExporter and regular exporters."""
        console_exporter = ConsoleSpanExporter()
        memory_exporter = InMemorySpanExporter()
        config = BasaltConfig(service_name="test-service")

        provider = create_tracer_provider(
            config, exporter=[console_exporter, memory_exporter]
        )

        # Verify both exporters were added
        self.assertIsInstance(provider, TracerProvider)
        self.assertEqual(len(provider._active_span_processor._span_processors), 2)

        # Test span export
        trace.set_tracer_provider(provider)
        tracer = trace.get_tracer("test")

        with tracer.start_as_current_span("test-span"):
            pass

        provider.force_flush()

        # Memory exporter should have received the span
        self.assertEqual(len(memory_exporter.get_finished_spans()), 1)

    def test_exporter_isolation_on_error(self):
        """Test that one failing exporter doesn't affect others."""
        failing_exporter = mock.Mock()
        failing_exporter.export.side_effect = Exception("Export failed")

        working_exporter = InMemorySpanExporter()
        config = BasaltConfig(service_name="test-service")

        provider = create_tracer_provider(
            config, exporter=[failing_exporter, working_exporter]
        )

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
        self.assertEqual(len(working_exporter.get_finished_spans()), 1)

    def test_duplicate_exporters_allowed(self):
        """Test that duplicate exporters in list are allowed (user responsibility)."""
        exporter = InMemorySpanExporter()
        config = BasaltConfig(service_name="test-service")

        # Same exporter instance twice
        provider = create_tracer_provider(config, exporter=[exporter, exporter])

        # Should have 2 processors (both using same exporter)
        self.assertEqual(len(provider._active_span_processor._span_processors), 2)

    def test_none_exporter_uses_console_with_warning(self):
        """Test that None exporter defaults to ConsoleSpanExporter with warning."""
        config = BasaltConfig(service_name="test-service")

        with self.assertWarns(UserWarning) as cm:
            provider = create_tracer_provider(config, exporter=None)

        # Verify warning message
        self.assertIn("No span exporter configured", str(cm.warning))

        # Verify provider was created
        self.assertIsInstance(provider, TracerProvider)


class TestTelemetryConfigWithMultipleExporters(unittest.TestCase):
    """Test TelemetryConfig with multiple exporters."""

    def test_config_accepts_exporter_list(self):
        """Test that TelemetryConfig accepts list of exporters."""
        exporter1 = InMemorySpanExporter()
        exporter2 = InMemorySpanExporter()

        config = TelemetryConfig(
            service_name="test-service",
            exporter=[exporter1, exporter2],
        )

        self.assertIsInstance(config.exporter, list)
        self.assertEqual(len(config.exporter), 2)
        self.assertIs(config.exporter[0], exporter1)
        self.assertIs(config.exporter[1], exporter2)

    def test_config_accepts_single_exporter(self):
        """Test backward compatibility: single exporter still works."""
        exporter = InMemorySpanExporter()

        config = TelemetryConfig(
            service_name="test-service",
            exporter=exporter,
        )

        # Should be the exporter itself, not wrapped in list
        self.assertIsInstance(config.exporter, InMemorySpanExporter)
        self.assertIs(config.exporter, exporter)

    def test_clone_with_exporter_list(self):
        """Test that clone() properly copies exporter lists."""
        exporter1 = InMemorySpanExporter()
        exporter2 = InMemorySpanExporter()

        original = TelemetryConfig(
            service_name="test-service",
            exporter=[exporter1, exporter2],
        )

        cloned = original.clone()

        # Verify it's a new list instance
        self.assertIsNot(cloned.exporter, original.exporter)
        # But contains same exporter objects
        self.assertEqual(len(cloned.exporter), 2)
        self.assertIs(cloned.exporter[0], exporter1)
        self.assertIs(cloned.exporter[1], exporter2)

    def test_clone_list_independence(self):
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
        self.assertEqual(len(original.exporter), 2)

    def test_clone_with_single_exporter(self):
        """Test that clone() handles single exporter correctly."""
        exporter = InMemorySpanExporter()

        original = TelemetryConfig(
            service_name="test-service",
            exporter=exporter,
        )

        cloned = original.clone()

        # Should be same exporter object (not cloned)
        self.assertIs(cloned.exporter, exporter)


if __name__ == "__main__":
    unittest.main()
