"""OpenTelemetry span processors for Basalt instrumentation."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from . import semconv
from .context_managers import (
    EVALUATOR_CONFIG_CONTEXT_KEY,
    EVALUATOR_CONTEXT_KEY,
    EVALUATOR_METADATA_CONTEXT_KEY,
    EvaluatorConfig,
    normalize_evaluator_specs,
)
from .trace_context import (
    ORGANIZATION_CONTEXT_KEY,
    USER_CONTEXT_KEY,
    TraceExperiment,
    TraceIdentity,
    _current_trace_defaults,
    _TraceContextConfig,
)

logger = logging.getLogger(__name__)


def _merge_evaluators(span: Span, slugs: Sequence[str]) -> None:
    """Merge evaluator slugs onto the span, avoiding duplicates."""

    if not slugs or not span.is_recording():
        return

    existing: list[str] = []
    attributes = getattr(span, "attributes", None)
    if isinstance(attributes, dict):
        current = attributes.get(semconv.BasaltSpan.EVALUATORS)
        if isinstance(current, (list, tuple)):
            existing.extend(str(value) for value in current if str(value).strip())

    merged: list[str] = []
    for slug in [*existing, *slugs]:
        if slug and slug not in merged:
            merged.append(slug)

    span.set_attribute(semconv.BasaltSpan.EVALUATORS, merged)


def _set_default_metadata(span: Span, defaults: _TraceContextConfig) -> None:
    if not span.is_recording():
        return

    # Only attach experiments to root spans (spans without a parent)
    # We check both span.parent (SpanContext) and our own internal tracking if needed
    parent_ctx = span.parent
    is_root_span = parent_ctx is None or (hasattr(parent_ctx, "is_valid") and not parent_ctx.is_valid)

    experiment = defaults.experiment if isinstance(defaults.experiment, TraceExperiment) else None
    if experiment and is_root_span:
        span.set_attribute(semconv.BasaltExperiment.ID, experiment.id)
        if experiment.name:
            span.set_attribute(semconv.BasaltExperiment.NAME, experiment.name)
        if experiment.feature_slug:
            span.set_attribute(semconv.BasaltExperiment.FEATURE_SLUG, experiment.feature_slug)

    for key, value in (defaults.observe_metadata or {}).items():
        span.set_attribute(f"{semconv.BASALT_META_PREFIX}{key}", value)


def _apply_user_org_from_context(span: Span, parent_context: Any | None = None) -> None:
    """Apply user and organization from OpenTelemetry context to the span."""
    if not span.is_recording():
        return

    # Read user from context
    user = otel_context.get_value(USER_CONTEXT_KEY, parent_context)
    if isinstance(user, TraceIdentity):
        span.set_attribute(semconv.BasaltUser.ID, user.id)
        if user.name:
            span.set_attribute(semconv.BasaltUser.NAME, user.name)

    # Read organization from context
    org = otel_context.get_value(ORGANIZATION_CONTEXT_KEY, parent_context)
    if isinstance(org, TraceIdentity):
        span.set_attribute(semconv.BasaltOrganization.ID, org.id)
        if org.name:
            span.set_attribute(semconv.BasaltOrganization.NAME, org.name)


class BasaltContextProcessor(SpanProcessor):
    """Apply Basalt trace defaults to every started span."""

    def on_start(self, span: Span, parent_context: Any | None = None) -> None:  # type: ignore[override]
        if not span.is_recording():
            return
        defaults = _current_trace_defaults()
        _set_default_metadata(span, defaults)
        # Apply user/org from OpenTelemetry context (enables propagation to child spans)
        _apply_user_org_from_context(span, parent_context)

    def on_end(self, span: ReadableSpan) -> None:  # type: ignore[override]
        return

    def shutdown(self) -> None:  # type: ignore[override]
        return

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # type: ignore[override]
        return True


class BasaltCallEvaluatorProcessor(SpanProcessor):
    """Attach call-scoped evaluators, config, and metadata discovered in the OTel context."""

    def __init__(self, context_key: str = EVALUATOR_CONTEXT_KEY) -> None:
        self._context_key = context_key

    def on_start(self, span: Span, parent_context: Any | None = None) -> None:  # type: ignore[override]
        if not span.is_recording():
            return

        # Attach evaluator slugs
        # Logic: Evaluators are span-scoped. They should attach to the *current* span being created
        # in this context. To prevent propagation to children, we rely on the fact that
        # the user should not nest operations that *both* pick up the same evaluator context
        # unless they intend to.
        # However, to strictly enforce "no propagation below", we would need to clear the context
        # or mark it as consumed. But we can't easily modify the context here.
        #
        # For now, we attach if present. The user's requirement "never propagate below"
        # is best handled by ensuring `with_evaluators` is used tightly around the target call.
        # If auto-instrumentation creates a span, it gets it.

        context_payload = otel_context.get_value(self._context_key, parent_context)
        if context_payload:
            raw: Iterable[Any]
            if isinstance(context_payload, (list, tuple, set)):
                raw = context_payload
            else:
                raw = [context_payload]

            try:
                attachments = normalize_evaluator_specs(list(raw))
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Failed to normalize call evaluators: %s", exc)
            else:
                slugs = [attachment.slug for attachment in attachments if attachment.slug]
                _merge_evaluators(span, slugs)

        # Attach evaluator config from context
        context_config = otel_context.get_value(EVALUATOR_CONFIG_CONTEXT_KEY, parent_context)
        if context_config and isinstance(context_config, EvaluatorConfig):
            try:
                import json
                span.set_attribute(semconv.BasaltSpan.EVALUATORS_CONFIG, json.dumps(context_config.to_dict()))
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Failed to set evaluator config: %s", exc)

        # Attach evaluator metadata from context
        context_metadata = otel_context.get_value(EVALUATOR_METADATA_CONTEXT_KEY, parent_context)
        if context_metadata and isinstance(context_metadata, dict):
            try:
                import json
                for key, value in context_metadata.items():
                    attr_key = f"{semconv.BasaltSpan.EVALUATOR_PREFIX}.metadata.{key}"
                    # Serialize value if needed
                    if value is None or isinstance(value, (str, bool, int, float)):
                        serialized = value
                    else:
                        try:
                            serialized = json.dumps(value)
                        except Exception:
                            serialized = str(value)
                    if serialized is not None:
                        span.set_attribute(attr_key, serialized)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Failed to set evaluator metadata: %s", exc)

    def on_end(self, span: ReadableSpan) -> None:  # type: ignore[override]
        return

    def shutdown(self) -> None:  # type: ignore[override]
        return

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # type: ignore[override]
        return True
