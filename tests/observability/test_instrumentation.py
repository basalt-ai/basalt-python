"""Tests for InstrumentationManager."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from basalt.observability.config import TelemetryConfig
from basalt.observability.instrumentation import InstrumentationManager


class TestInstrumentationManager(unittest.TestCase):
    @mock.patch("basalt.observability.instrumentation.setup_tracing")
    def test_initialize_disabled_skips_tracing(self, mock_setup):
        manager = InstrumentationManager()

        manager.initialize(TelemetryConfig(enabled=False))

        mock_setup.assert_not_called()

    @mock.patch("basalt.observability.instrumentation.setup_tracing")
    def test_initialize_enables_tracing(self, mock_setup):
        mock_setup.return_value = mock.Mock()
        manager = InstrumentationManager()

        manager.initialize(TelemetryConfig(service_name="svc"))

        mock_setup.assert_called_once()

    @mock.patch.object(InstrumentationManager, "_uninstrument_providers")
    def test_shutdown_flushes_provider(self, mock_uninstrument):
        manager = InstrumentationManager()
        manager._initialized = True
        provider = mock.Mock()
        manager._tracer_provider = provider

        manager.shutdown()

        mock_uninstrument.assert_called_once()
        provider.force_flush.assert_called_once()
        provider.shutdown.assert_called_once()
        self.assertFalse(manager._initialized)

    @mock.patch.object(InstrumentationManager, "_instrument_providers")
    def test_initialize_instrumentation_sets_env_and_instruments_providers(self, mock_providers):
        config = TelemetryConfig(
            trace_content=False,
            enabled_providers=["openai", "anthropic"],
        )
        manager = InstrumentationManager()

        with mock.patch.dict(os.environ, {}, clear=False):
            manager._initialize_instrumentation(config)
            self.assertEqual(os.environ["TRACELOOP_TRACE_CONTENT"], "false")

        mock_providers.assert_called_once_with(config)

    @mock.patch.dict(
        os.environ,
        {"BASALT_OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4317"},
        clear=False,
    )
    @mock.patch("basalt.observability.instrumentation.OTLPSpanExporter")
    def test_build_exporter_from_env_adds_bearer_for_grpc(self, mock_grpc_exporter):
        mock_grpc_exporter.return_value = mock.Mock()
        manager = InstrumentationManager()
        manager._resolve_api_key("test-key")

        exporter = manager._build_exporter_from_env()

        self.assertIs(exporter, mock_grpc_exporter.return_value)
        mock_grpc_exporter.assert_called_once()
        headers = mock_grpc_exporter.call_args.kwargs["headers"]
        self.assertEqual(headers["authorization"], "Bearer test-key")

    @mock.patch.dict(
        os.environ,
        {
            "BASALT_OTEL_EXPORTER_OTLP_ENDPOINT": "https://collector/v1/traces",
            "BASALT_API_KEY": "env-key",
        },
        clear=False,
    )
    @mock.patch("basalt.observability.instrumentation.OTLPHTTPSpanExporter")
    @mock.patch("basalt.observability.instrumentation.OTLPSpanExporter")
    def test_build_exporter_from_env_adds_bearer_for_http(
        self,
        mock_grpc_exporter,
        mock_http_exporter,
    ):
        mock_http_exporter.return_value = mock.Mock()
        manager = InstrumentationManager()

        exporter = manager._build_exporter_from_env()

        self.assertIs(exporter, mock_http_exporter.return_value)
        mock_http_exporter.assert_called_once()
        mock_grpc_exporter.assert_not_called()
        headers = mock_http_exporter.call_args.kwargs["headers"]
        self.assertEqual(headers["authorization"], "Bearer env-key")

    def test_should_instrument_provider_default(self):
        """Test that by default all providers are instrumented."""
        config = TelemetryConfig()
        self.assertTrue(config.should_instrument_provider("openai"))
        self.assertTrue(config.should_instrument_provider("anthropic"))
        self.assertTrue(config.should_instrument_provider("langchain"))

    def test_should_instrument_provider_with_enabled_list(self):
        """Test that only enabled providers are instrumented when specified."""
        config = TelemetryConfig(enabled_providers=["openai", "anthropic"])
        self.assertTrue(config.should_instrument_provider("openai"))
        self.assertTrue(config.should_instrument_provider("anthropic"))
        self.assertFalse(config.should_instrument_provider("langchain"))
        self.assertFalse(config.should_instrument_provider("llamaindex"))

    def test_should_instrument_provider_with_disabled_list(self):
        """Test that disabled providers are not instrumented."""
        config = TelemetryConfig(disabled_providers=["langchain", "llamaindex"])
        self.assertTrue(config.should_instrument_provider("openai"))
        self.assertTrue(config.should_instrument_provider("anthropic"))
        self.assertFalse(config.should_instrument_provider("langchain"))
        self.assertFalse(config.should_instrument_provider("llamaindex"))

    def test_should_instrument_provider_disabled_takes_precedence(self):
        """Test that disabled list takes precedence over enabled list."""
        config = TelemetryConfig(
            enabled_providers=["openai", "anthropic"],
            disabled_providers=["anthropic"],
        )
        self.assertTrue(config.should_instrument_provider("openai"))
        self.assertFalse(config.should_instrument_provider("anthropic"))

    @mock.patch("basalt.observability.instrumentation._safe_import")
    def test_instrument_providers_respects_config(self, mock_import):
        """Test that _instrument_providers respects the configuration."""
        # Mock instrumentor with is_instrumented_by_opentelemetry property
        mock_instrumentor = mock.Mock()
        mock_instrumentor.is_instrumented_by_opentelemetry = False  # Not yet instrumented
        mock_instrumentor_cls = mock.Mock(return_value=mock_instrumentor)

        # Only return instrumentor class for openai and anthropic
        def safe_import_side_effect(module, name):
            if "openai" in module:
                return mock_instrumentor_cls
            elif "anthropic" in module:
                return mock_instrumentor_cls
            return None

        mock_import.side_effect = safe_import_side_effect

        config = TelemetryConfig(enabled_providers=["openai", "anthropic"])
        manager = InstrumentationManager()
        manager._instrument_providers(config)

        # Should have called instrument() for both providers
        self.assertEqual(mock_instrumentor.instrument.call_count, 2)
        self.assertIn("openai", manager._provider_instrumentors)
        self.assertIn("anthropic", manager._provider_instrumentors)
