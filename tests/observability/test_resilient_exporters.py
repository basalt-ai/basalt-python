"""Tests for resilient exporter wrapper."""

from __future__ import annotations

from unittest import mock

import pytest
from opentelemetry.sdk.trace.export import SpanExportResult

from basalt.observability.resilient_exporters import ResilientSpanExporter


def test_successful_export_delegates_to_underlying_exporter():
    """When export succeeds, return the result from underlying exporter."""
    mock_exporter = mock.Mock()
    mock_exporter.export.return_value = SpanExportResult.SUCCESS

    wrapper = ResilientSpanExporter(mock_exporter)
    spans = [mock.Mock()]

    result = wrapper.export(spans)

    assert result == SpanExportResult.SUCCESS
    mock_exporter.export.assert_called_once_with(spans)


def test_connection_error_returns_failure_without_raising():
    """When connection error occurs, log and return FAILURE."""
    mock_exporter = mock.Mock()
    mock_exporter.export.side_effect = ConnectionError("DNS resolution failed")

    wrapper = ResilientSpanExporter(mock_exporter)
    spans = [mock.Mock()]

    # Should not raise
    result = wrapper.export(spans)

    assert result == SpanExportResult.FAILURE


@mock.patch("basalt.observability.resilient_exporters.logger")
def test_exception_logged_at_warning_level(mock_logger):
    """Verify exceptions are logged at warning level."""
    mock_exporter = mock.Mock()
    error = ConnectionError("Connection refused")
    mock_exporter.export.side_effect = error

    wrapper = ResilientSpanExporter(mock_exporter)
    wrapper.export([mock.Mock()])

    # Should log at warning level so users see the error
    mock_logger.warning.assert_called_once()
    # Check that it was called with the format string and exception type/message
    call_args = mock_logger.warning.call_args[0]
    assert "Span export failed" in call_args[0]
    assert call_args[1] == "ConnectionError"
    assert call_args[2] == error


@pytest.mark.parametrize(
    "exception",
    [
        ConnectionError("Connection failed"),
        OSError("Network unreachable"),
        TimeoutError("Request timeout"),
        RuntimeError("Unexpected error"),
    ],
)
def test_various_exception_types_caught(exception):
    """Verify different exception types are all caught."""
    mock_exporter = mock.Mock()
    mock_exporter.export.side_effect = exception

    wrapper = ResilientSpanExporter(mock_exporter)

    # Should not raise
    result = wrapper.export([mock.Mock()])
    assert result == SpanExportResult.FAILURE


def test_shutdown_suppresses_exceptions():
    """Shutdown errors are suppressed and logged."""
    mock_exporter = mock.Mock()
    mock_exporter.shutdown.side_effect = RuntimeError("Shutdown failed")

    wrapper = ResilientSpanExporter(mock_exporter)

    # Should not raise
    wrapper.shutdown()

    mock_exporter.shutdown.assert_called_once()


def test_force_flush_suppresses_exceptions_returns_false():
    """Force flush errors return False."""
    mock_exporter = mock.Mock()
    mock_exporter.force_flush.side_effect = RuntimeError("Flush failed")

    wrapper = ResilientSpanExporter(mock_exporter)

    result = wrapper.force_flush(1000)

    assert result is False
    mock_exporter.force_flush.assert_called_once_with(1000)


def test_force_flush_success_returns_true():
    """Force flush success returns True."""
    mock_exporter = mock.Mock()
    mock_exporter.force_flush.return_value = True

    wrapper = ResilientSpanExporter(mock_exporter)

    result = wrapper.force_flush(5000)

    assert result is True
    mock_exporter.force_flush.assert_called_once_with(5000)


def test_custom_exception_types_can_be_specified():
    """Can configure which exception types to suppress."""
    mock_exporter = mock.Mock()
    mock_exporter.export.side_effect = ValueError("Not a connection error")

    # Only suppress ConnectionError
    wrapper = ResilientSpanExporter(
        mock_exporter,
        suppress_exceptions=(ConnectionError,),
    )

    # Should raise ValueError since it's not in suppress list
    with pytest.raises(ValueError):
        wrapper.export([mock.Mock()])


def test_custom_exception_types_suppress_configured_types():
    """Custom exception types work correctly."""
    mock_exporter = mock.Mock()
    mock_exporter.export.side_effect = ConnectionError("Connection failed")

    # Only suppress ConnectionError
    wrapper = ResilientSpanExporter(
        mock_exporter,
        suppress_exceptions=(ConnectionError,),
    )

    # Should not raise ConnectionError
    result = wrapper.export([mock.Mock()])
    assert result == SpanExportResult.FAILURE
