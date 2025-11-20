"""Client-level telemetry integration tests."""

from __future__ import annotations

import unittest
from unittest import mock

from basalt.client import Basalt
from basalt.observability.config import TelemetryConfig
from basalt.observability.instrumentation import InstrumentationManager


class TestBasaltClientTelemetry(unittest.TestCase):
    @mock.patch.object(InstrumentationManager, "initialize")
    def test_enable_telemetry_false_disables_config(self, mock_initialize):
        client = Basalt(api_key="key", enable_telemetry=False)

        self.assertTrue(mock_initialize.called)
        config_arg = mock_initialize.call_args[0][0]
        self.assertFalse(config_arg.enabled)
        self.assertEqual(mock_initialize.call_args.kwargs["api_key"], "key")

        client.shutdown()

    @mock.patch.object(InstrumentationManager, "shutdown")
    @mock.patch.object(InstrumentationManager, "initialize")
    def test_shutdown_invokes_instrumentation(self, mock_initialize, mock_shutdown):
        client = Basalt(api_key="key")

        client.shutdown()

        mock_initialize.assert_called_once()
        mock_shutdown.assert_called_once()

    @mock.patch.object(InstrumentationManager, "initialize")
    def test_custom_telemetry_config_passed_through(self, mock_initialize):
        telemetry = TelemetryConfig(service_name="custom")

        Basalt(api_key="key", telemetry_config=telemetry)

        mock_initialize.assert_called_once_with(telemetry, api_key="key")

