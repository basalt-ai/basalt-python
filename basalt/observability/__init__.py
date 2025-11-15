"""Observability facade for the Basalt SDK."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from .config import TelemetryConfig
from .context_managers import (
    EvaluatorConfig,
    EventSpanHandle,
    FunctionSpanHandle,
    LLMSpanHandle,
    RetrievalSpanHandle,
    SpanHandle,
    ToolSpanHandle,
    trace_event,
    trace_function,
    trace_generation,
    trace_retrieval,
    trace_span,
    trace_tool,
    with_evaluators,
)
from .decorators import (
    ObserveKind,
    evaluator,
    observe,
    observe_event,
    observe_function,
    observe_generation,
    observe_retrieval,
    observe_span,
    observe_tool,
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

__all__ = [
    "TelemetryConfig",
    "InstrumentationManager",
    "evaluator",
    # New observe API
    "observe",
    "ObserveKind",
    "observe_span",
    "observe_generation",
    "observe_retrieval",
    "observe_function",
    "observe_tool",
    "observe_event",
    # Context managers (keeping trace_* names)
    "trace_span",
    "trace_generation",
    "trace_retrieval",
    "trace_function",
    "trace_tool",
    "trace_event",
    # Deprecated decorators (for backward compatibility)
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
    "EvaluatorConfig",
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
    "set_span_attribute",
    "set_span_attributes",
    "add_span_event",
    "record_span_exception",
    "set_span_status",
    "set_span_status_ok",
    "set_span_status_error",
    "get_current_span",
    "get_current_span_handle",
    # Newly added helpers
    "get_parent_span",
    "get_trace",
    "get_otel_trace",
]

_instrumentation = InstrumentationManager()


def configure_trace_defaults(
    *,
    config: TraceContextConfig | None = None,
    experiment: TraceExperiment | dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    evaluators: Iterable[str] | None = None,
) -> TraceContextConfig:
    """
    Configure global defaults applied to newly created spans.

    Note: User and organization are now set at the span level, not globally.
    Use the user/organization parameters on context managers or decorators instead.

    Args:
        config: Optional base configuration to extend.
        experiment: Default experiment metadata to attach.
        metadata: Arbitrary metadata key/values added to spans.
        evaluators: Evaluator slugs automatically attached to spans.

    Returns:
        The effective configuration that will be applied going forward.
    """
    base = config.clone() if config else TraceContextConfig()
    payload = TraceContextConfig(
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
    """
    Set user identity for the current span and propagate to child spans.

    This sets user attributes on the active span and attaches the user identity
    to the OpenTelemetry context, ensuring child spans inherit it.

    Args:
        user_id: Unique identifier for the user
        name: Optional display name for the user
    """
    from opentelemetry.context import attach, set_value

    from .trace_context import USER_CONTEXT_KEY, TraceIdentity

    # Set on current span
    span = current_span()
    if span:
        span.set_attribute("basalt.user.id", user_id)
        if name:
            span.set_attribute("basalt.user.name", name)

    # Propagate to child spans via context
    user_identity = TraceIdentity(id=user_id, name=name)
    attach(set_value(USER_CONTEXT_KEY, user_identity))


def set_trace_organization(organization_id: str, name: str | None = None) -> None:
    """
    Set organization identity for the current span and propagate to child spans.

    This sets organization attributes on the active span and attaches the organization
    identity to the OpenTelemetry context, ensuring child spans inherit it.

    Args:
        organization_id: Unique identifier for the organization
        name: Optional display name for the organization
    """
    from opentelemetry.context import attach, set_value

    from .trace_context import ORGANIZATION_CONTEXT_KEY, TraceIdentity

    # Set on current span
    span = current_span()
    if span:
        span.set_attribute("basalt.organization.id", organization_id)
        if name:
            span.set_attribute("basalt.organization.name", name)

    # Propagate to child spans via context
    org_identity = TraceIdentity(id=organization_id, name=name)
    attach(set_value(ORGANIZATION_CONTEXT_KEY, org_identity))


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


def get_current_span() -> Span | None:  # Lightweight alias
    """Return the active OpenTelemetry span if valid, else None.

    Provided as a user-friendly alias so callers don't have to import
    opentelemetry.trace themselves.
    """
    return current_span()


def get_current_span_handle() -> SpanHandle | None:  # Lightweight alias
    """Return a SpanHandle wrapper for the active span if available."""
    return current_span_handle()


def set_span_attribute(key: str, value: Any) -> bool:
    """Set a single attribute on the current span.

    Returns True if the attribute was set, False if no active span.
    """
    span = current_span()
    if not span:
        return False
    span.set_attribute(key, value)
    return True


def set_span_attributes(attributes: Mapping[str, Any] | dict[str, Any]) -> int:
    """Set multiple attributes on the current span.

    Returns the number of attributes applied. Returns 0 if no active span.
    """
    span = current_span()
    if not span:
        return 0
    count = 0
    for k, v in (attributes or {}).items():
        span.set_attribute(k, v)
        count += 1
    return count


def add_span_event(name: str, attributes: Mapping[str, Any] | dict[str, Any] | None = None) -> bool:
    """Add an event to the current span.

    Returns True if the event was added, False if no active span.
    """
    span = current_span()
    if not span:
        return False
    span.add_event(name, attributes=dict(attributes) if isinstance(attributes, Mapping) else attributes)
    return True


def record_span_exception(exc: BaseException) -> bool:
    """Record an exception on the current span.

    Returns True if recorded, False if no active span.
    """
    span = current_span()
    if not span:
        return False
    span.record_exception(exc)
    return True


def set_span_status(status_code: StatusCode, description: str | None = None) -> bool:
    """Set the status of the current span.

    Returns True if status was set, False if no active span.
    """
    span = current_span()
    if not span:
        return False
    span.set_status(Status(status_code, description))
    return True


def set_span_status_ok(description: str | None = None) -> bool:
    """Convenience to set StatusCode.OK on the current span."""
    return set_span_status(StatusCode.OK, description)


def set_span_status_error(description: str | None = None) -> bool:
    """Convenience to set StatusCode.ERROR on the current span."""
    return set_span_status(StatusCode.ERROR, description)

# TODO : Get Parent span (get_parent_span()) / Get Otel Trace (get_otel_trace())
# TODO: Get Top/Root span (get_trace())

# ---------------------------------------------------------------------------
# Span hierarchy helpers
# ---------------------------------------------------------------------------

def _get_span_parent(span: Span) -> Span | None:
    """Attempt to retrieve the parent span object if still in memory.

    The OpenTelemetry Python SDK does not expose a public API to fetch the
    parent *Span object* from a child span – only the parent span context is
    guaranteed to be available. However, when the active parent span is still
    on the context stack (typical while you are inside nested Basalt span
    context managers) SDK span implementations usually retain a ``parent``
    attribute we can inspect. This function best‑effort looks for that.
    """
    # First, prefer the explicit back-reference Basalt attaches on creation
    parent = getattr(span, "_basalt_parent_span", None)
    if parent and getattr(parent, "get_span_context", None):
        ctx = parent.get_span_context()
        if ctx and ctx.is_valid:
            return parent  # type: ignore[return-value]
    # Fallback: some SDK span implementations keep a `.parent` attribute
    try:  # Best effort – guard against differing SDK implementations
        parent = getattr(span, "parent", None)
        if parent and getattr(parent, "get_span_context", None):
            ctx = parent.get_span_context()
            if ctx and ctx.is_valid:
                return parent  # type: ignore[return-value]
    except Exception:  # pragma: no cover - defensive
        return None
    return None


def get_parent_span() -> Span | None:
    """Return the parent span of the current active span, if available.

    Returns None if there is no active span, the active span is invalid,
    or its parent span object cannot be resolved (e.g. the parent already
    finished and is no longer on the context stack).
    """
    span = current_span()
    if not span:
        return None
    return _get_span_parent(span)


def get_trace() -> Span | None:
    """Return the root (top-most) span of the current trace.

    Walks up the in-memory parent chain (if available) until no further
    parent can be resolved. If there is no active span, returns None.
    """
    span = current_span()
    if not span:
        return None
    root = span
    visited = set()
    while True:
        # Protect against potential cycles (extremely unlikely)
        ident = id(root)
        if ident in visited:
            break
        visited.add(ident)
        parent = _get_span_parent(root)
        if not parent:
            break
        root = parent
    return root


def get_otel_trace():
    """Return the OpenTelemetry SpanContext for the current span, if valid.

    Returns None if there is no active span or its context is invalid.
    """
    span = current_span()
    if not span:
        return None
    ctx = span.get_span_context()
    if not ctx or not ctx.is_valid:
        return None
    return ctx

# ---------------------------------------------------------------------------
