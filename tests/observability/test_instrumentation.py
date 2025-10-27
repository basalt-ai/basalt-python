"""Tests for InstrumentationManager."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from basalt.observability.config import TelemetryConfig
from basalt.observability.instrumentation import InstrumentationManager


class TestInstrumentationManager(unittest.TestCase):
    @mock.patch.object(InstrumentationManager, "_instrument_http")
    @mock.patch("basalt.observability.instrumentation.setup_tracing")
    def test_initialize_disabled_skips_tracing(self, mock_setup, mock_http):
        manager = InstrumentationManager()

        manager.initialize(TelemetryConfig(enabled=False))

        mock_setup.assert_not_called()
        mock_http.assert_not_called()

    @mock.patch.object(InstrumentationManager, "_instrument_http")
    @mock.patch("basalt.observability.instrumentation.setup_tracing")
    def test_initialize_enables_tracing(self, mock_setup, mock_http):
        mock_setup.return_value = mock.Mock()
        manager = InstrumentationManager()

        manager.initialize(TelemetryConfig(service_name="svc"))

        mock_setup.assert_called_once()
        mock_http.assert_called_once()

    @mock.patch.object(InstrumentationManager, "_uninstrument_llm")
    @mock.patch.object(InstrumentationManager, "_uninstrument_http")
    def test_shutdown_flushes_provider(self, mock_http, mock_llm):
        manager = InstrumentationManager()
        manager._initialized = True
        provider = mock.Mock()
        manager._tracer_provider = provider

        manager.shutdown()

        mock_http.assert_called_once()
        mock_llm.assert_called_once()
        provider.force_flush.assert_called_once()
        provider.shutdown.assert_called_once()
        self.assertFalse(manager._initialized)

    @mock.patch.object(InstrumentationManager, "_instrument_llm_providers")
    def test_initialize_llm_instrumentation_sets_env_and_instruments_providers(self, mock_providers):
        config = TelemetryConfig(
            llm_trace_content=False,
            llm_enabled_providers=["openai", "anthropic"],
        )
        manager = InstrumentationManager()

        with mock.patch.dict(os.environ, {}, clear=False):
            manager._initialize_llm_instrumentation(config)
            self.assertEqual(os.environ["TRACELOOP_TRACE_CONTENT"], "false")

        mock_providers.assert_called_once_with(config)

    def test_should_instrument_provider_default(self):
        """Test that by default all providers are instrumented."""
        config = TelemetryConfig()
        self.assertTrue(config.should_instrument_provider("openai"))
        self.assertTrue(config.should_instrument_provider("anthropic"))
        self.assertTrue(config.should_instrument_provider("langchain"))

    def test_should_instrument_provider_with_enabled_list(self):
        """Test that only enabled providers are instrumented when specified."""
        config = TelemetryConfig(llm_enabled_providers=["openai", "anthropic"])
        self.assertTrue(config.should_instrument_provider("openai"))
        self.assertTrue(config.should_instrument_provider("anthropic"))
        self.assertFalse(config.should_instrument_provider("langchain"))
        self.assertFalse(config.should_instrument_provider("llamaindex"))

    def test_should_instrument_provider_with_disabled_list(self):
        """Test that disabled providers are not instrumented."""
        config = TelemetryConfig(llm_disabled_providers=["langchain", "llamaindex"])
        self.assertTrue(config.should_instrument_provider("openai"))
        self.assertTrue(config.should_instrument_provider("anthropic"))
        self.assertFalse(config.should_instrument_provider("langchain"))
        self.assertFalse(config.should_instrument_provider("llamaindex"))

    def test_should_instrument_provider_disabled_takes_precedence(self):
        """Test that disabled list takes precedence over enabled list."""
        config = TelemetryConfig(
            llm_enabled_providers=["openai", "anthropic"],
            llm_disabled_providers=["anthropic"],
        )
        self.assertTrue(config.should_instrument_provider("openai"))
        self.assertFalse(config.should_instrument_provider("anthropic"))

    @mock.patch("basalt.observability.instrumentation._safe_import")
    def test_instrument_llm_providers_respects_config(self, mock_import):
        """Test that _instrument_llm_providers respects the configuration."""
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

        config = TelemetryConfig(llm_enabled_providers=["openai", "anthropic"])
        manager = InstrumentationManager()
        manager._instrument_llm_providers(config)

        # Should have called instrument() for both providers
        self.assertEqual(mock_instrumentor.instrument.call_count, 2)
        self.assertIn("openai", manager._provider_instrumentors)
        self.assertIn("anthropic", manager._provider_instrumentors)
