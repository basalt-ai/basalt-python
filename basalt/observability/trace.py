from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from opentelemetry.trace import Span, Tracer

from .context_managers import (
    get_current_span,
    get_tracer,
)


class Trace:
    """
    Low-level tracing primitives for advanced users.
    Provides direct access to OpenTelemetry objects.
    """

    @staticmethod
    def current_span() -> Span | None:
        """Get the current OpenTelemetry span."""
        return get_current_span()

    @staticmethod
    def get_tracer(name: str = "basalt.custom") -> Tracer:
        """Get an OpenTelemetry tracer."""
        return get_tracer(name)

    @staticmethod
    def add_event(name: str, attributes: Mapping[str, Any] | None = None) -> None:
        """Add a raw event to the current span."""
        span = get_current_span()
        if span:
            span.add_event(name, attributes=attributes)

    @staticmethod
    def set_attribute(key: str, value: Any) -> None:
        """Set a raw attribute on the current span."""
        span = get_current_span()
        if span:
            span.set_attribute(key, value)

    @staticmethod
    def set_attributes(attributes: Mapping[str, Any]) -> None:
        """Set multiple raw attributes on the current span."""
        span = get_current_span()
        if span:
            span.set_attributes(attributes)

# Singleton instance
trace_api = Trace
