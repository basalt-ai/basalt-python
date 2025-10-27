"""Tests for InstrumentationManager."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from basalt.observability.config import OpenLLMetryConfig, TelemetryConfig
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
    @mock.patch("basalt.observability.instrumentation._safe_import")
    def test_initialize_openllmetry_sets_env_and_inits_traceloop(self, mock_import, mock_providers):
        traceloop = mock.Mock()
        mock_import.side_effect = lambda module, name: traceloop if module == "traceloop.sdk" else None
        config = OpenLLMetryConfig(
            app_name="app",
            disable_batch=True,
            trace_content=False,
            telemetry_enabled=False,
            api_endpoint="https://example.com",
            headers={"Authorization": "token"},
        )
        manager = InstrumentationManager()

        with mock.patch.dict(os.environ, {}, clear=False):
            manager._initialize_openllmetry(config)
            self.assertEqual(os.environ["TRACELOOP_TRACE_CONTENT"], "false")
            self.assertEqual(os.environ["TRACELOOP_TELEMETRY"], "false")
        traceloop.init.assert_called_once()
        mock_providers.assert_called_once_with()
