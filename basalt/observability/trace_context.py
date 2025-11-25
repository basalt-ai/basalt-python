"""Utilities to manage default trace context for observability spans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from threading import RLock
from typing import Any, Final

from opentelemetry import context as otel_context
from opentelemetry.trace import Span

from . import semconv

# Context keys for user and organization propagation
USER_CONTEXT_KEY: Final[str] = "basalt.context.user"
ORGANIZATION_CONTEXT_KEY: Final[str] = "basalt.context.organization"
FEATURE_SLUG_CONTEXT_KEY: Final[str] = "basalt.context.feature_slug"


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
class _TraceContextConfig:
    """
    Internal configuration for trace defaults.
    Not exposed publicly.
    """

    experiment: TraceExperiment | str | None = None
    observe_metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.experiment = _coerce_experiment(self.experiment)
        self.observe_metadata = dict(self.observe_metadata) if self.observe_metadata else {}

    def clone(self) -> _TraceContextConfig:
        """Return a defensive copy of the configuration."""
        return _TraceContextConfig(
            experiment=self.experiment,
            observe_metadata=dict(self.observe_metadata) if self.observe_metadata is not None else {},
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


def _coerce_experiment(payload: TraceExperiment | str | None) -> TraceExperiment | None:
    if payload is None or isinstance(payload, TraceExperiment):
        return payload
    if isinstance(payload, str):
        if not payload:
            raise ValueError("Trace experiment ID must be a non-empty string.")
        return TraceExperiment(id=payload, name=None, feature_slug=None)
    # Check for Experiment dataclass (avoid isinstance to prevent circular import)
    if hasattr(payload, "id"):
        exp_id = payload.id
        if not isinstance(exp_id, str) or not exp_id:
            raise ValueError("Experiment ID must be a non-empty string.")
        exp_name = getattr(payload, "name", None)
        exp_feature_slug = getattr(payload, "feature_slug", None)
        return TraceExperiment(id=exp_id, name=exp_name, feature_slug=exp_feature_slug)
    raise TypeError("Trace experiment must be a string, Experiment object, or TraceExperiment.")


_DEFAULT_CONTEXT: _TraceContextConfig = _TraceContextConfig()
_LOCK = RLock()


def _set_trace_defaults(config: _TraceContextConfig | None) -> None:
    """Replace the globally configured trace defaults (Internal)."""
    global _DEFAULT_CONTEXT
    with _LOCK:
        _DEFAULT_CONTEXT = config.clone() if config else _TraceContextConfig()


def _current_trace_defaults() -> _TraceContextConfig:
    """Return a clone of the currently configured trace defaults (Internal)."""
    with _LOCK:
        return _DEFAULT_CONTEXT.clone()


def configure_global_metadata(metadata: dict[str, Any] | None) -> None:
    """
    Configure global observability metadata applied to all traces.

    Args:
        metadata: Dictionary of metadata key-value pairs.
    """
    config = _TraceContextConfig(observe_metadata=metadata)
    _set_trace_defaults(config)


def apply_trace_defaults(span: Span, defaults: _TraceContextConfig | None = None) -> None:
    """Attach the configured defaults to the provided span."""
    context = defaults.clone() if defaults else _current_trace_defaults()

    # Experiments are attached to root spans only (checked by processor usually, but good to have helper)
    if isinstance(context.experiment, TraceExperiment):
        # Note: The processor calling this should check for root span if strict adherence is needed here,
        # but we'll set attributes and let the processor decide or we check here.
        # For now, we just set attributes. The processor `_set_default_metadata` handles the root check.
        span.set_attribute(semconv.BasaltExperiment.ID, context.experiment.id)
        if context.experiment.name:
            span.set_attribute(semconv.BasaltExperiment.NAME, context.experiment.name)
        if context.experiment.feature_slug:
            span.set_attribute(semconv.BasaltExperiment.FEATURE_SLUG, context.experiment.feature_slug)

    if context.observe_metadata:
        for key, value in context.observe_metadata.items():
            span.set_attribute(f"{semconv.BASALT_META_PREFIX}{key}", value)


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
