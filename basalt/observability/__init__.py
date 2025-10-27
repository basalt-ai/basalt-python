"""Observability facade for the Basalt SDK."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Span

from .config import TelemetryConfig
from .context_managers import (
    EventSpanHandle,
    LLMSpanHandle,
    RetrievalSpanHandle,
    SpanHandle,
    ToolSpanHandle,
    trace_event,
    trace_llm_call,
    trace_retrieval,
    trace_span,
    trace_tool,
)
from .decorators import trace_http, trace_llm, trace_operation
from .instrumentation import InstrumentationManager
from .trace_context import (
    TraceContextConfig,
    TraceExperiment,
    TraceIdentity,
    current_trace_defaults,
    set_trace_defaults,
    update_default_evaluators,
)

__all__ = [
    "TelemetryConfig",
    "InstrumentationManager",
    "trace_operation",
    "trace_llm",
    "trace_http",
    "trace_span",
    "trace_llm_call",
    "trace_retrieval",
    "trace_tool",
    "trace_event",
    "SpanHandle",
    "LLMSpanHandle",
    "RetrievalSpanHandle",
    "ToolSpanHandle",
    "EventSpanHandle",
    "TraceContextConfig",
    "TraceIdentity",
    "TraceExperiment",
    "configure_trace_defaults",
    "clear_trace_defaults",
    "get_trace_defaults",
    "add_default_evaluators",
    "init",
    "observe",
    "observe_cm",
    "Observation",
    "current_span",
    "set_trace_user",
    "set_trace_organization",
    "set_trace_session",
    "set_trace_env",
    "add_trace_tags",
    "add_trace_metadata",
    "attach_trace_experiment",
    "add_span_evaluator",
    "flush",
]

_instrumentation = InstrumentationManager()


def configure_trace_defaults(
    *,
    config: TraceContextConfig | None = None,
    user: TraceIdentity | dict[str, Any] | None = None,
    organization: TraceIdentity | dict[str, Any] | None = None,
    experiment: TraceExperiment | dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    evaluators: Iterable[str] | None = None,
) -> TraceContextConfig:
    """
    Configure global defaults applied to newly created spans.

    Args:
        config: Optional base configuration to extend.
        user: Default user identity to attach.
        organization: Default organization identity to attach.
        experiment: Default experiment metadata to attach.
        metadata: Arbitrary metadata key/values added to spans.
        evaluators: Evaluator slugs automatically attached to spans.

    Returns:
        The effective configuration that will be applied going forward.
    """
    base = config.clone() if config else TraceContextConfig()
    payload = TraceContextConfig(
        user=user if user is not None else base.user,
        organization=organization if organization is not None else base.organization,
        experiment=experiment if experiment is not None else base.experiment,
        metadata=metadata if metadata is not None else {str(k): v for k, v in (base.metadata or {}).items()},
        evaluators=list(evaluators) if evaluators is not None else list(base.evaluators or []),
    )
    set_trace_defaults(payload)
    return current_trace_defaults()


def clear_trace_defaults() -> None:
    """Clear any globally configured trace defaults."""
    set_trace_defaults(None)


def get_trace_defaults() -> TraceContextConfig:
    """Return the currently active trace defaults."""
    return current_trace_defaults()


def add_default_evaluators(*evaluators: str) -> TraceContextConfig:
    """Append evaluator slugs to the default trace configuration."""
    update_default_evaluators([slug for slug in evaluators if slug])
    return current_trace_defaults()


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
    telemetry_config = TelemetryConfig(
        service_name=app_name,
        environment=environment,
        exporter=exporter,
        enable_llm_instrumentation=enable_openllmetry,
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

    def set_user(self, user_id: str, name: str | None = None) -> None:
        self._span.set_attribute("basalt.user.id", user_id)
        if name:
            self._span.set_attribute("basalt.user.name", name)

    def set_organization(self, organization_id: str, name: str | None = None) -> None:
        self._span.set_attribute("basalt.organization.id", organization_id)
        if name:
            self._span.set_attribute("basalt.organization.name", name)

    def set_session(self, session_id: str) -> None:
        self._span.set_attribute("basalt.session.id", session_id)

    def set_environment(self, environment: str) -> None:
        self._span.set_attribute("deployment.environment", environment)

    def add_tags(self, tags: Iterable[str]) -> None:
        self._span.set_attribute("basalt.trace.tags", list(tags))

    def add_metadata(self, metadata: dict[str, Any]) -> None:
        for key, value in metadata.items():
            self._span.set_attribute(f"basalt.meta.{key}", value)

    def add_evaluator(self, evaluator_slug: str) -> None:
        evaluators = _collect_evaluators(self._span)
        if evaluator_slug not in evaluators:
            evaluators.append(evaluator_slug)
        self._span.set_attribute("basalt.trace.evaluators", evaluators)

    def set_experiment(
        self,
        experiment_id: str,
        *,
        name: str | None = None,
        feature_slug: str | None = None,
    ) -> None:
        self._span.set_attribute("basalt.experiment.id", experiment_id)
        if name:
            self._span.set_attribute("basalt.experiment.name", name)
        if feature_slug:
            self._span.set_attribute("basalt.experiment.feature_slug", feature_slug)


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


def _collect_evaluators(span: Span) -> list[str]:
    existing = getattr(span, "attributes", None)
    if isinstance(existing, dict):
        evaluators = existing.get("basalt.trace.evaluators")
        if isinstance(evaluators, (list, tuple)):
            return [item for item in evaluators if isinstance(item, str)]
    return []


def current_span() -> Span | None:
    span = trace.get_current_span()
    return span if span and span.get_span_context().is_valid else None


def set_trace_user(user_id: str, name: str | None = None) -> None:
    span = current_span()
    if span:
        span.set_attribute("basalt.user.id", user_id)
        if name:
            span.set_attribute("basalt.user.name", name)


def set_trace_organization(organization_id: str, name: str | None = None) -> None:
    span = current_span()
    if span:
        span.set_attribute("basalt.organization.id", organization_id)
        if name:
            span.set_attribute("basalt.organization.name", name)


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


def attach_trace_experiment(
    experiment_id: str,
    *,
    name: str | None = None,
    feature_slug: str | None = None,
) -> None:
    span = current_span()
    if not span:
        return
    span.set_attribute("basalt.experiment.id", experiment_id)
    if name:
        span.set_attribute("basalt.experiment.name", name)
    if feature_slug:
        span.set_attribute("basalt.experiment.feature_slug", feature_slug)


def add_span_evaluator(evaluator_slug: str) -> None:
    span = current_span()
    if not span or not evaluator_slug:
        return
    evaluators = _collect_evaluators(span)
    if evaluator_slug not in evaluators:
        evaluators.append(evaluator_slug)
    span.set_attribute("basalt.trace.evaluators", evaluators)


def flush() -> None:
    """Force flush span processors without shutting down the provider."""
    provider = trace.get_tracer_provider()
    try:
        provider.force_flush()  # type: ignore[attr-defined]
    except Exception:
        pass
