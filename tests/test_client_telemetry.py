"""Client-level telemetry integration tests."""

from __future__ import annotations

import unittest
from unittest import mock

from basalt.client import Basalt
from basalt.observability.config import TelemetryConfig
from basalt.observability.instrumentation import InstrumentationManager
from basalt.observability.trace_context import TraceContextConfig


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

    @mock.patch("basalt.client.configure_trace_defaults")
    @mock.patch.object(InstrumentationManager, "initialize")
    def test_constructor_configures_trace_defaults(self, mock_initialize, mock_configure):
        Basalt(
            api_key="key",
            trace_user={"id": "user-1", "name": "Jane"},
            trace_metadata={"env": "test"},
            trace_evaluators=["eval-1", "eval-2"],
        )

        mock_configure.assert_called_once_with(
            user={"id": "user-1", "name": "Jane"},
            metadata={"env": "test"},
            evaluators=["eval-1", "eval-2"],
        )

    @mock.patch("basalt.client.configure_trace_defaults")
    @mock.patch.object(InstrumentationManager, "initialize")
    def test_trace_context_overrides_apply(self, mock_initialize, mock_configure):
        context = TraceContextConfig(
            user={"id": "base-user"},
            organization={"id": "org-1"},
            metadata={"env": "prod"},
            evaluators=["eval-base"],
        )

        Basalt(
            api_key="key",
            trace_context=context,
            trace_organization={"id": "org-2", "name": "Org"},
            trace_experiment={"id": "exp-1", "feature_slug": "slug"},
        )

        mock_configure.assert_called_once()
        _, kwargs = mock_configure.call_args
        self.assertEqual(kwargs["user"].id, "base-user")
        self.assertIsNone(kwargs["user"].name)
        self.assertEqual(kwargs["organization"], {"id": "org-2", "name": "Org"})
        self.assertEqual(kwargs["experiment"], {"id": "exp-1", "feature_slug": "slug"})
        self.assertEqual(kwargs["metadata"], {"env": "prod"})
        self.assertEqual(kwargs["evaluators"], ["eval-base"])
