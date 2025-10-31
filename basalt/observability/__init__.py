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
    FunctionSpanHandle,
    LLMSpanHandle,
    RetrievalSpanHandle,
    SpanHandle,
    ToolSpanHandle,
    trace_event,
    trace_function,
    trace_generation,
    trace_llm_call,
    trace_retrieval,
    trace_span,
    trace_tool,
    with_evaluators,
)
from .decorators import (
    evaluator,
    trace_operation,
)
from .decorators import (
    trace_event as trace_event_decorator,
)
from .decorators import (
    trace_function as trace_function_decorator,
)
from .decorators import (
    trace_generation as trace_generation_decorator,
)
from .decorators import (
    trace_llm as trace_llm_decorator,
)
from .decorators import (
    trace_retrieval as trace_retrieval_decorator,
)
from .decorators import (
    trace_span as trace_span_decorator,
)
from .decorators import (
    trace_tool as trace_tool_decorator,
)
from .evaluators import (
    attach_evaluator,
    attach_evaluators_to_current_span,
    attach_evaluators_to_span,
)
from .instrumentation import InstrumentationManager
from .processors import BasaltCallEvaluatorProcessor, BasaltContextProcessor
from .trace_context import (
    TraceContextConfig,
    TraceExperiment,
    TraceIdentity,
    current_trace_defaults,
    set_trace_defaults,
    update_default_evaluators,
)

trace_llm = trace_llm_decorator

__all__ = [
    "TelemetryConfig",
    "InstrumentationManager",
    "trace_operation",
    "trace_llm",
    "trace_llm_decorator",
    "evaluator",
    "trace_span",
    "trace_generation",
    "trace_llm_call",
    "trace_retrieval",
    "trace_function",
    "trace_tool",
    "trace_event",
    "trace_span_decorator",
    "trace_generation_decorator",
    "trace_retrieval_decorator",
    "trace_function_decorator",
    "trace_tool_decorator",
    "trace_event_decorator",
    "SpanHandle",
    "LLMSpanHandle",
    "RetrievalSpanHandle",
    "ToolSpanHandle",
    "FunctionSpanHandle",
    "EventSpanHandle",
    "TraceContextConfig",
    "TraceIdentity",
    "TraceExperiment",
    "BasaltContextProcessor",
    "BasaltCallEvaluatorProcessor",
    "with_evaluators",
    "configure_trace_defaults",
    "clear_trace_defaults",
    "get_trace_defaults",
    "add_default_evaluators",
    "init",
    "observe",
    "observe_cm",
    "Observation",
    "current_span",
    "current_span_handle",
    "update_current_span",
    "set_trace_user",
    "set_trace_organization",
    "set_trace_session",
    "set_trace_env",
    "add_trace_tags",
    "add_trace_metadata",
    "attach_trace_experiment",
    "add_span_evaluator",
    "flush",
    # Evaluator functions
    "attach_evaluator",
    "attach_evaluators_to_span",
    "attach_evaluators_to_current_span",
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


def add_default_evaluators(*evaluators: Any) -> TraceContextConfig:
    """Append evaluator specs to the default trace configuration."""
    update_default_evaluators([spec for spec in evaluators if spec])
    return current_trace_defaults()


def init(
    app_name: str = "basalt-sdk",
    *,
    environment: str | None = None,
    exporter: Any | None = None,
    enable_openllmetry: bool = False,
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

    def add_evaluator(self, evaluator: Any) -> None:
        self._handle.add_evaluators(evaluator)

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


def current_span() -> Span | None:
    span = trace.get_current_span()
    return span if span and span.get_span_context().is_valid else None


def current_span_handle() -> SpanHandle | None:
    """Return a SpanHandle wrapper for the active span, if any."""
    span = current_span()
    if not span:
        return None
    return SpanHandle(span)


def update_current_span(
    *,
    input_payload: Any | None = None,
    output_payload: Any | None = None,
    variables: dict[str, Any] | None = None,
    evaluators: Iterable[Any] | None = None,
) -> SpanHandle | None:
    """
    Update the active span with IO payloads and evaluators.

    Returns:
        The span handle if a span was active, otherwise None.
    """
    handle = current_span_handle()
    if not handle:
        return None
    if input_payload is not None:
        handle.set_input(input_payload)
    if output_payload is not None:
        handle.set_output(output_payload)
    if variables is not None:
        handle.set_variables(variables)
    if evaluators:
        handle.add_evaluators(*evaluators)
    return handle


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
    """
    Attaches experiment metadata to the current trace span.
    Parameters:
        experiment_id (str): The unique identifier of the experiment.
        name (str | None, optional): The name of the experiment. If provided, it is added as a span attribute.
        feature_slug (str | None, optional): The feature slug associated with the experiment.
             If provided, it is added as a span attribute.
    Returns:
        None
    Notes:
        If there is no current span, the function returns without attaching any attributes.
    """
    span = current_span()
    if not span:
        return
    span.set_attribute("basalt.experiment.id", experiment_id)
    if name:
        span.set_attribute("basalt.experiment.name", name)
    if feature_slug:
        span.set_attribute("basalt.experiment.feature_slug", feature_slug)


def add_span_evaluator(evaluator: Any) -> None:
    """Attach an evaluator to the current span."""
    if not evaluator:
        return
    update_current_span(evaluators=[evaluator])


def flush() -> None:
    """Force flush span processors without shutting down the provider."""
    provider = trace.get_tracer_provider()
    try:
        provider.force_flush()  # type: ignore[attr-defined]
    except Exception:
        pass
