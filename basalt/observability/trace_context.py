"""Utilities to manage default trace context for observability spans."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from threading import RLock
from typing import Any, Final

from opentelemetry import context as otel_context
from opentelemetry.trace import Span

from . import semconv

# Context keys for user and organization propagation
USER_CONTEXT_KEY: Final[str] = "basalt.context.user"
ORGANIZATION_CONTEXT_KEY: Final[str] = "basalt.context.organization"


@dataclass(frozen=True, slots=True)
class TraceIdentity:
    """Identity payload attached to traces (users, organizations)."""

    id: str
    name: str | None = None


@dataclass(frozen=True, slots=True)
class TraceExperiment:
    """Experiment metadata associated with a trace."""

    id: str
    name: str | None = None
    feature_slug: str | None = None


@dataclass(slots=True)
class TraceContextConfig:
    """Default attributes applied to newly created spans."""

    experiment: TraceExperiment | Mapping[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    evaluators: list[Any] | None = None

    def __post_init__(self) -> None:
        self.experiment = _coerce_experiment(self.experiment)
        self.metadata = dict(self.metadata) if self.metadata else {}
        self.evaluators = list(self.evaluators) if self.evaluators else []

    def clone(self) -> TraceContextConfig:
        """Return a defensive copy of the configuration."""
        return TraceContextConfig(
            experiment=self.experiment,
            metadata=dict(self.metadata) if self.metadata is not None else {},
            evaluators=list(self.evaluators) if self.evaluators is not None else [],
        )


def _coerce_identity(payload: TraceIdentity | Mapping[str, Any] | None) -> TraceIdentity | None:
    if payload is None or isinstance(payload, TraceIdentity):
        return payload
    if not isinstance(payload, Mapping):
        raise TypeError("Trace identity must be a mapping or TraceIdentity.")
    identifier = payload.get("id")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("Trace identity mapping requires a non-empty 'id'.")
    name = payload.get("name")
    if name is not None and not isinstance(name, str):
        raise ValueError("Trace identity 'name' must be a string.")
    return TraceIdentity(id=identifier, name=name)


def _coerce_experiment(payload: TraceExperiment | Mapping[str, Any] | None) -> TraceExperiment | None:
    if payload is None or isinstance(payload, TraceExperiment):
        return payload
    if not isinstance(payload, Mapping):
        raise TypeError("Trace experiment must be a mapping or TraceExperiment.")
    identifier = payload.get("id")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("Trace experiment mapping requires a non-empty 'id'.")
    name = payload.get("name")
    if name is not None and not isinstance(name, str):
        raise ValueError("Trace experiment 'name' must be a string.")
    feature_slug = payload.get("feature_slug")
    if feature_slug is not None and not isinstance(feature_slug, str):
        raise ValueError("Trace experiment 'feature_slug' must be a string.")
    return TraceExperiment(id=identifier, name=name, feature_slug=feature_slug)


_DEFAULT_CONTEXT: TraceContextConfig = TraceContextConfig()
_LOCK = RLock()


def set_trace_defaults(config: TraceContextConfig | None) -> None:
    """Replace the globally configured trace defaults."""
    global _DEFAULT_CONTEXT
    with _LOCK:
        _DEFAULT_CONTEXT = config.clone() if config else TraceContextConfig()


def current_trace_defaults() -> TraceContextConfig:
    """Return a clone of the currently configured trace defaults."""
    with _LOCK:
        return _DEFAULT_CONTEXT.clone()


def apply_trace_defaults(span: Span, defaults: TraceContextConfig | None = None) -> None:
    """Attach the configured defaults to the provided span."""
    context = defaults.clone() if defaults else current_trace_defaults()
    if isinstance(context.experiment, TraceExperiment):
        span.set_attribute(semconv.BasaltExperiment.ID, context.experiment.id)
        if context.experiment.name:
            span.set_attribute(semconv.BasaltExperiment.NAME, context.experiment.name)
        if context.experiment.feature_slug:
            span.set_attribute(semconv.BasaltExperiment.FEATURE_SLUG, context.experiment.feature_slug)
    if context.metadata:
        for key, value in context.metadata.items():
            span.set_attribute(f"{semconv.BASALT_META_PREFIX}{key}", value)


def update_default_evaluators(new_evaluators: Iterable[Any]) -> None:
    """Add evaluators to the default context without replacing the configuration."""
    global _DEFAULT_CONTEXT
    with _LOCK:
        merged = list(_DEFAULT_CONTEXT.evaluators) if _DEFAULT_CONTEXT.evaluators is not None else []
        for spec in new_evaluators:
            if spec not in merged:
                merged.append(spec)
        _DEFAULT_CONTEXT = replace(_DEFAULT_CONTEXT, evaluators=merged)


def get_context_user() -> TraceIdentity | None:
    """Retrieve user identity from the current OpenTelemetry context."""
    return otel_context.get_value(USER_CONTEXT_KEY)


def get_context_organization() -> TraceIdentity | None:
    """Retrieve organization identity from the current OpenTelemetry context."""
    return otel_context.get_value(ORGANIZATION_CONTEXT_KEY)


def apply_user_from_context(span: Span, user: TraceIdentity | Mapping[str, Any] | None = None) -> None:
    """
    Apply user identity to a span from the provided value or OpenTelemetry context.

    Args:
        span: The span to apply user identity to.
        user: Optional user identity. If None, retrieves from context.
    """
    if user is not None:
        user_identity = _coerce_identity(user)
    else:
        user_identity = get_context_user()

    if isinstance(user_identity, TraceIdentity):
        span.set_attribute(semconv.BasaltUser.ID, user_identity.id)
        if user_identity.name:
            span.set_attribute(semconv.BasaltUser.NAME, user_identity.name)


def apply_organization_from_context(
    span: Span, organization: TraceIdentity | Mapping[str, Any] | None = None
) -> None:
    """
    Apply organization identity to a span from the provided value or OpenTelemetry context.

    Args:
        span: The span to apply organization identity to.
        organization: Optional organization identity. If None, retrieves from context.
    """
    if organization is not None:
        org_identity = _coerce_identity(organization)
    else:
        org_identity = get_context_organization()

    if isinstance(org_identity, TraceIdentity):
        span.set_attribute(semconv.BasaltOrganization.ID, org_identity.id)
        if org_identity.name:
            span.set_attribute(semconv.BasaltOrganization.NAME, org_identity.name)
