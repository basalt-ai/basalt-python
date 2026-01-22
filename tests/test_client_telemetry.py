"""Client-level telemetry integration tests."""

from __future__ import annotations

from unittest import mock

from basalt.client import Basalt
from basalt.observability.config import TelemetryConfig
from basalt.observability.instrumentation import InstrumentationManager


def test_enable_telemetry_false_disables_config(monkeypatch):
    mock_initialize = mock.Mock()
    monkeypatch.setattr(InstrumentationManager, "initialize", mock_initialize)

    client = Basalt(api_key="key", enable_telemetry=False)

    assert mock_initialize.called
    config_arg = mock_initialize.call_args[0][0]
    assert not config_arg.enabled
    assert mock_initialize.call_args.kwargs["api_key"] == "key"

    client.shutdown()


def test_shutdown_invokes_instrumentation(monkeypatch):
    mock_initialize = mock.Mock()
    mock_shutdown = mock.Mock()
    monkeypatch.setattr(InstrumentationManager, "initialize", mock_initialize)
    monkeypatch.setattr(InstrumentationManager, "shutdown", mock_shutdown)

    client = Basalt(api_key="key")

    client.shutdown()

    mock_initialize.assert_called_once()
    mock_shutdown.assert_called_once()


def test_custom_telemetry_config_passed_through(monkeypatch):
    mock_initialize = mock.Mock()
    monkeypatch.setattr(InstrumentationManager, "initialize", mock_initialize)

    telemetry = TelemetryConfig(service_name="custom")

    Basalt(api_key="key", telemetry_config=telemetry)

    mock_initialize.assert_called_once_with(telemetry, api_key="key")
