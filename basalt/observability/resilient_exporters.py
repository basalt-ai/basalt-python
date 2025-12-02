"""Resilient wrappers for OpenTelemetry span exporters."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

logger = logging.getLogger(__name__)


class ResilientSpanExporter(SpanExporter):
    """
    Wrapper around a SpanExporter that catches and logs connection errors.

    This wrapper prevents connection errors (DNS failures, timeouts, etc.)
    from propagating as exceptions during span export. Instead, errors are
    logged at warning level and export is marked as failed without raising.

    This is particularly useful for HTTP exporters where the endpoint may
    be temporarily unavailable or misconfigured. The BatchSpanProcessor
    will continue trying to export subsequent batches.

    Args:
        exporter: The underlying SpanExporter to wrap
        suppress_exceptions: Tuple of exception types to suppress (default: all Exception)
    """

    def __init__(
        self,
        exporter: SpanExporter,
        suppress_exceptions: tuple[type[Exception], ...] | None = None,
    ) -> None:
        self._exporter = exporter
        # Default to suppressing all exceptions, but allow customization
        self._suppress_exceptions = suppress_exceptions or (Exception,)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """
        Export spans with error suppression.

        Catches connection errors and other exceptions during export,
        logs them at warning level, and returns FAILURE result instead
        of letting exceptions propagate.
        """
        try:
            return self._exporter.export(spans)
        except self._suppress_exceptions as exc:
            # Log at warning level so users see connection errors without stacktraces
            logger.warning(
                "Span export failed (endpoint may be unavailable): %s: %s",
                type(exc).__name__,
                exc,
            )
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        """Shutdown the underlying exporter."""
        try:
            self._exporter.shutdown()
        except Exception:
            logger.debug("Error during exporter shutdown", exc_info=True)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush the underlying exporter."""
        try:
            return self._exporter.force_flush(timeout_millis)
        except Exception:
            logger.debug("Error during exporter force_flush", exc_info=True)
            return False
