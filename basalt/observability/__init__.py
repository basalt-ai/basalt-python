"""Observability facade for the Basalt SDK."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Span

from .config import OpenLLMetryConfig, TelemetryConfig
from .context_managers import SpanHandle, trace_llm_call, trace_retrieval, trace_span
from .decorators import trace_http, trace_llm, trace_operation
from .instrumentation import InstrumentationManager

__all__ = [
    "TelemetryConfig",
    "OpenLLMetryConfig",
    "InstrumentationManager",
    "trace_operation",
    "trace_llm",
    "trace_http",
    "trace_span",
    "trace_llm_call",
    "trace_retrieval",
    "init",
    "observe",
    "observe_cm",
    "Observation",
    "current_span",
    "set_trace_user",
    "set_trace_session",
    "set_trace_env",
    "add_trace_tags",
    "add_trace_metadata",
    "flush",
]

_instrumentation = InstrumentationManager()


def init(
    app_name: str = "basalt-sdk",
    *,
    environment: str | None = None,
    exporter: Any | None = None,
    enable_openllmetry: bool = False,
    instrument_http: bool = True,
) -> None:
    """Deprecated façade around InstrumentationManager.initialize."""
    warnings.warn(
        "basalt.observability.init() is deprecated. "
        "Pass telemetry_config to Basalt() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    openll_config = None
    if enable_openllmetry:
        openll_config = OpenLLMetryConfig(
            app_name=app_name,
        )

    telemetry_config = TelemetryConfig(
        service_name=app_name,
        environment=environment,
        exporter=exporter,
        enable_openllmetry=enable_openllmetry,
        openllmetry_config=openll_config,
        instrument_http=instrument_http,
    )
    _instrumentation.initialize(telemetry_config)


class Observation:
    """Helper to operate on a span handle for backward compatibility."""

    def __init__(self, handle: SpanHandle):
        self._handle = handle
        self._span = handle.span

    def add_attributes(self, attrs: dict[str, Any]) -> None:
        for key, value in (attrs or {}).items():
            self._span.set_attribute(key, value)

    def event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self._span.add_event(name=name, attributes=attributes)

    def set_user(self, user_id: str) -> None:
        self._span.set_attribute("basalt.user.id", user_id)

    def set_session(self, session_id: str) -> None:
        self._span.set_attribute("basalt.session.id", session_id)

    def set_environment(self, environment: str) -> None:
        self._span.set_attribute("deployment.environment", environment)

    def add_tags(self, tags: Iterable[str]) -> None:
        self._span.set_attribute("basalt.trace.tags", list(tags))

    def add_metadata(self, metadata: dict[str, Any]) -> None:
        for key, value in metadata.items():
            self._span.set_attribute(f"basalt.meta.{key}", value)


def observe(
    name: str | None = None,
    *,
    attributes: dict[str, Any] | Callable[..., dict[str, Any]] | None = None,
    capture_io: bool = False,
):
    """Deprecated alias for trace_operation."""
    warnings.warn(
        "basalt.observability.observe is deprecated; use trace_operation instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return trace_operation(name=name, attributes=attributes, capture_io=capture_io)


@contextmanager
def observe_cm(name: str, attributes: dict[str, Any] | None = None):
    """Deprecated context manager alias around trace_span."""
    warnings.warn(
        "basalt.observability.observe_cm is deprecated; use trace_span instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    with trace_span(name, attributes=attributes) as handle:
        yield Observation(handle)


def current_span() -> Span | None:
    span = trace.get_current_span()
    return span if span and span.get_span_context().is_valid else None


def set_trace_user(user_id: str) -> None:
    span = current_span()
    if span:
        span.set_attribute("basalt.user.id", user_id)


def set_trace_session(session_id: str) -> None:
    span = current_span()
    if span:
        span.set_attribute("basalt.session.id", session_id)


def set_trace_env(environment: str) -> None:
    span = current_span()
    if span:
        span.set_attribute("deployment.environment", environment)


def add_trace_tags(tags: Iterable[str]) -> None:
    span = current_span()
    if span:
        span.set_attribute("basalt.trace.tags", list(tags))


def add_trace_metadata(metadata: dict[str, Any]) -> None:
    span = current_span()
    if span:
        for key, value in metadata.items():
            span.set_attribute(f"basalt.meta.{key}", value)


def flush() -> None:
    """Force flush span processors without shutting down the provider."""
    provider = trace.get_tracer_provider()
    try:
        provider.force_flush()  # type: ignore[attr-defined]
    except Exception:
        pass
