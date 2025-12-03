"""OpenTelemetry span processors for Basalt instrumentation."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from typing import Any, Final

from opentelemetry import context as otel_context
from opentelemetry.context import attach
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from . import semconv
from .context_managers import (
    EVALUATOR_CONFIG_CONTEXT_KEY,
    EVALUATOR_CONTEXT_KEY,
    EvaluationConfig,
    normalize_evaluator_specs,
)
from .decorators import ObserveKind
from .trace_context import (
    ORGANIZATION_CONTEXT_KEY,
    USER_CONTEXT_KEY,
    TraceExperiment,
    TraceIdentity,
    _current_trace_defaults,
    _TraceContextConfig,
)

logger = logging.getLogger(__name__)

# Context keys for pending auto-instrumentation injection data
PENDING_INJECT_INPUT_KEY: Final[str] = "basalt.pending_inject.input"
PENDING_INJECT_OUTPUT_KEY: Final[str] = "basalt.pending_inject.output"
PENDING_INJECT_VARIABLES_KEY: Final[str] = "basalt.pending_inject.variables"
PENDING_INJECT_METADATA_KEY: Final[str] = "basalt.pending_inject.metadata"
PENDING_INJECT_PROMPT_KEY: Final[str] = "basalt.pending_inject.prompt"


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
        normalized = str(slug).strip()
        if normalized and normalized not in merged:
            merged.append(normalized)

    span.set_attribute(semconv.BasaltSpan.EVALUATORS, merged)


def _set_default_metadata(span: Span, defaults: _TraceContextConfig) -> None:
    if not span.is_recording():
        return

    # Only attach experiments to root spans (spans without a parent)
    # We check both span.parent (SpanContext) and our own internal tracking if needed
    parent_ctx = span.parent
    is_root_span = parent_ctx is None or (
        hasattr(parent_ctx, "is_valid") and not parent_ctx.is_valid
    )

    experiment = (
        defaults.experiment
        if isinstance(defaults.experiment, TraceExperiment)
        else None
    )
    if experiment and is_root_span:
        span.set_attribute(semconv.BasaltExperiment.ID, experiment.id)
        if experiment.name:
            span.set_attribute(semconv.BasaltExperiment.NAME, experiment.name)
        if experiment.feature_slug:
            span.set_attribute(
                semconv.BasaltExperiment.FEATURE_SLUG, experiment.feature_slug
            )

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


def _apply_feature_slug_from_context(span: Span, parent_context: Any | None = None) -> None:
    """Apply feature slug from OpenTelemetry context to the span."""
    if not span.is_recording():
        return

    from .trace_context import FEATURE_SLUG_CONTEXT_KEY

    # Read feature_slug from context
    feature_slug = otel_context.get_value(FEATURE_SLUG_CONTEXT_KEY, parent_context)

    if isinstance(feature_slug, str) and feature_slug.strip():
        span.set_attribute(semconv.BasaltSpan.FEATURE_SLUG, feature_slug)
        logger.debug(f"Set feature_slug={feature_slug} on span={span.name}")


class BasaltContextProcessor(SpanProcessor):
    """Apply Basalt trace defaults to every started span."""

    def on_start(self, span: Span, parent_context: Any | None = None) -> None:  # type: ignore[override]
        if not span.is_recording():
            return
        defaults = _current_trace_defaults()
        _set_default_metadata(span, defaults)
        # Apply user/org from OpenTelemetry context (enables propagation to child spans)
        _apply_user_org_from_context(span, parent_context)
        # Apply feature_slug from OpenTelemetry context (enables propagation to child spans)
        _apply_feature_slug_from_context(span, parent_context)

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
                slugs = [
                    attachment.slug for attachment in attachments if attachment.slug
                ]
                _merge_evaluators(span, slugs)

        # Attach evaluator config from context
        context_config = otel_context.get_value(
            EVALUATOR_CONFIG_CONTEXT_KEY, parent_context
        )
        if context_config and isinstance(context_config, EvaluationConfig):
            try:
                import json

                span.set_attribute(
                    semconv.BasaltSpan.EVALUATION_CONFIG,
                    json.dumps(context_config.to_dict()),
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Failed to set evaluator config: %s", exc)

    def on_end(self, span: ReadableSpan) -> None:  # type: ignore[override]
        return

    def shutdown(self) -> None:  # type: ignore[override]
        return

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # type: ignore[override]
        return True


# Known auto-instrumentation scope names
KNOWN_AUTO_INSTRUMENTATION_SCOPES: Final[frozenset[str]] = frozenset({
    "opentelemetry.instrumentation.openai",
    "opentelemetry.instrumentation.openai.v1",  # OpenAI SDK v1+
    "opentelemetry.instrumentation.anthropic",
    "opentelemetry.instrumentation.google_genai",
    "opentelemetry.instrumentation.google_generativeai",
    "opentelemetry.instrumentation.bedrock",
    "opentelemetry.instrumentation.vertexai",
    "opentelemetry.instrumentation.ollama",
    "opentelemetry.instrumentation.mistralai",
    "opentelemetry.instrumentation.langchain",
    "opentelemetry.instrumentation.llamaindex",
    "opentelemetry.instrumentation.chromadb",
    "opentelemetry.instrumentation.pinecone",
    "opentelemetry.instrumentation.qdrant",
})


# Mapping of instrumentation scope names to Basalt span kinds
# This determines which basalt.span.kind attribute is set for auto-instrumented spans
INSTRUMENTATION_SCOPE_KINDS: Final[dict[str, str]] = {
    # LLM Providers - all map to ObserveKind.GENERATION
    "opentelemetry.instrumentation.openai": ObserveKind.GENERATION.value,
    "opentelemetry.instrumentation.openai.v1": ObserveKind.GENERATION.value,  # OpenAI SDK v1+
    "opentelemetry.instrumentation.anthropic": ObserveKind.GENERATION.value,
    "opentelemetry.instrumentation.google_genai": ObserveKind.GENERATION.value,
    "opentelemetry.instrumentation.google_generativeai": ObserveKind.GENERATION.value,
    "opentelemetry.instrumentation.bedrock": ObserveKind.GENERATION.value,
    "opentelemetry.instrumentation.vertexai": ObserveKind.GENERATION.value,
    "opentelemetry.instrumentation.ollama": ObserveKind.GENERATION.value,
    "opentelemetry.instrumentation.mistralai": ObserveKind.GENERATION.value,
    # Vector Databases - all map to ObserveKind.RETRIEVAL
    "opentelemetry.instrumentation.chromadb": ObserveKind.RETRIEVAL.value,
    "opentelemetry.instrumentation.pinecone": ObserveKind.RETRIEVAL.value,
    "opentelemetry.instrumentation.qdrant": ObserveKind.RETRIEVAL.value,
    # Frameworks - these may create spans of various kinds, so we don't auto-assign
    # Users can manually set kind via observe decorators if needed
    # "opentelemetry.instrumentation.langchain": None,
    # "opentelemetry.instrumentation.llamaindex": None,
}


class BasaltAutoInstrumentationProcessor(SpanProcessor):
    """
    Span processor that injects pending data into auto-instrumented spans.

    This processor detects spans created by auto-instrumentation libraries
    (OpenAI, Anthropic, LangChain, etc.) and applies data that was previously
    stored via Observe.inject_for_auto_instrumentation(). The data is cleared
    after being applied to ensure single-use semantics.
    """

    def on_start(self, span: Span, parent_context: Any | None = None) -> None:  # type: ignore[override]
        """
        Called when a span starts.

        Detects auto-instrumented spans and applies pending injection data.
        Also automatically sets basalt.span.kind based on the instrumentation scope.
        """
        # Check if this is a recording span
        if not span.is_recording():
            return

        # Get instrumentation scope
        scope = span.instrumentation_scope
        if not scope or scope.name not in KNOWN_AUTO_INSTRUMENTATION_SCOPES:
            return

        # Automatically set span kind based on instrumentation scope
        span_kind = INSTRUMENTATION_SCOPE_KINDS.get(scope.name)
        if span_kind:
            span.set_attribute(semconv.BasaltSpan.KIND, span_kind)

        # Mark auto-instrumented spans within a basalt trace
        from .context_managers import ROOT_SPAN_CONTEXT_KEY

        if otel_context.get_value(ROOT_SPAN_CONTEXT_KEY) is not None:
            span.set_attribute(semconv.BasaltSpan.IN_TRACE, True)

        # Get context
        ctx = parent_context or otel_context.get_current()

        # Read and apply input
        input_payload = otel_context.get_value(PENDING_INJECT_INPUT_KEY, ctx)
        if input_payload is not None:
            serialized = json.dumps(input_payload) if not isinstance(input_payload, str) else input_payload
            span.set_attribute(semconv.BasaltSpan.INPUT, serialized)

        # Read and apply output
        output_payload = otel_context.get_value(PENDING_INJECT_OUTPUT_KEY, ctx)
        if output_payload is not None:
            serialized = json.dumps(output_payload) if not isinstance(output_payload, str) else output_payload
            span.set_attribute(semconv.BasaltSpan.OUTPUT, serialized)

        # Read and apply variables
        variables = otel_context.get_value(PENDING_INJECT_VARIABLES_KEY, ctx)
        if variables:
            span.set_attribute(semconv.BasaltSpan.VARIABLES, json.dumps(variables))

        # Read and apply metadata
        metadata = otel_context.get_value(PENDING_INJECT_METADATA_KEY, ctx)
        if metadata:
            from .utils import apply_span_metadata
            if isinstance(metadata, dict):
                apply_span_metadata(span, metadata)
            else:
                # Non-dict metadata: store at basalt.metadata
                span.set_attribute(semconv.BasaltSpan.METADATA, json.dumps(metadata) if not isinstance(metadata, str) else metadata)

        # Read and apply prompt metadata
        prompt_data = otel_context.get_value(PENDING_INJECT_PROMPT_KEY, ctx)
        if prompt_data:
            # Apply prompt attributes (slug, version, provider, model)
            for key, value in prompt_data.items():
                span.set_attribute(f"basalt.prompt.{key}", value)
        else:
            # If no explicit injection, try to read from ContextVar
            # This allows auto-instrumented spans to inherit prompt context
            # from the parent prompt context manager
            try:
                from basalt.prompts.models import _current_prompt_context
                prompt_ctx = _current_prompt_context.get()
                if prompt_ctx:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.debug(f"✓ Injecting prompt context from ContextVar for span '{scope.name}': slug='{prompt_ctx['slug']}'")
                    # Inject prompt attributes from ContextVar
                    span.set_attribute("basalt.prompt.slug", prompt_ctx["slug"])
                    if prompt_ctx.get("version"):
                        span.set_attribute("basalt.prompt.version", prompt_ctx["version"])
                    if prompt_ctx.get("tag"):
                        span.set_attribute("basalt.prompt.tag", prompt_ctx["tag"])
                    span.set_attribute("basalt.prompt.model.provider", prompt_ctx["provider"])
                    span.set_attribute("basalt.prompt.model.model", prompt_ctx["model"])
                    if prompt_ctx.get("variables"):
                        span.set_attribute("basalt.prompt.variables", json.dumps(prompt_ctx["variables"]))
                    span.set_attribute("basalt.prompt.from_cache", prompt_ctx["from_cache"])
                else:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.debug(f"✗ No prompt context found in ContextVar for auto-instrumented span '{scope.name}'")
            except (ImportError, LookupError) as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"✗ Failed to read prompt context for span '{scope.name}': {e}")
                # Prompts module not available or no context set - skip injection
                pass

        # Clear all pending injection data (single-use semantics)
        # We create a new context with all injection keys set to None
        new_ctx = ctx
        new_ctx = otel_context.set_value(PENDING_INJECT_INPUT_KEY, None, new_ctx)
        new_ctx = otel_context.set_value(PENDING_INJECT_OUTPUT_KEY, None, new_ctx)
        new_ctx = otel_context.set_value(PENDING_INJECT_VARIABLES_KEY, None, new_ctx)
        new_ctx = otel_context.set_value(PENDING_INJECT_METADATA_KEY, None, new_ctx)
        new_ctx = otel_context.set_value(PENDING_INJECT_PROMPT_KEY, None, new_ctx)
        attach(new_ctx)

    def on_end(self, span: ReadableSpan) -> None:  # type: ignore[override]
        return

    def shutdown(self) -> None:  # type: ignore[override]
        return

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # type: ignore[override]
        return True
