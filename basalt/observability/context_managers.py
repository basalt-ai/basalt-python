"""Context manager utilities for manual telemetry spans."""

from __future__ import annotations

import json
import logging
import os
import warnings
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Final

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.context import attach, detach, set_value
from opentelemetry.trace import Span, Status, StatusCode, Tracer

from . import semconv
from .trace_context import (
    ORGANIZATION_CONTEXT_KEY,
    USER_CONTEXT_KEY,
    TraceContextConfig,
    TraceIdentity,
    apply_organization_from_context,
    apply_user_from_context,
    current_trace_defaults,
)

SPAN_TYPE_ATTRIBUTE = semconv.BasaltSpan.TYPE
EVALUATOR_CONTEXT_KEY: Final[str] = "basalt.context.evaluators"
EVALUATOR_CONFIG_CONTEXT_KEY: Final[str] = "basalt.context.evaluator_config"
EVALUATOR_METADATA_CONTEXT_KEY: Final[str] = "basalt.context.evaluator_metadata"
logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class EvaluatorConfig:
    """
    Type-safe configuration for evaluators attached to a span.

    This configuration is span-scoped and shared by all evaluators in the span.
    It's not handled client-side but attached to the span for server-side processing.

    Attributes:
        sample_rate: Sampling rate for evaluators (0.0-1.0). Default is 1.0 (100%).
    """

    sample_rate: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.sample_rate <= 1.0:
            raise ValueError("sample_rate must be within [0.0, 1.0].")

    def to_dict(self) -> dict[str, Any]:
        """Convert config to a dictionary for serialization."""
        return {"sample_rate": self.sample_rate}


@dataclass(slots=True)
class EvaluatorAttachment:
    """Normalized evaluator payload applied to spans."""

    slug: str
    metadata: Mapping[str, Any] | None = field(default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.slug, str) or not self.slug.strip():
            raise ValueError("Evaluator slug must be a non-empty string.")
        self.slug = self.slug.strip()
        if self.metadata is not None and not isinstance(self.metadata, Mapping):
            raise TypeError("Evaluator metadata must be a mapping.")


def _normalize_evaluator_entry(entry: Any) -> EvaluatorAttachment:
    """Convert assorted evaluator payloads into EvaluatorAttachment objects."""
    if isinstance(entry, EvaluatorAttachment):
        return entry
    if isinstance(entry, str):
        return EvaluatorAttachment(slug=entry)
    if isinstance(entry, Mapping):
        payload = dict(entry)
        slug = payload.pop("slug", None)
        if slug is None:
            raise ValueError("Evaluator mapping must include a 'slug' key.")
        metadata = payload.pop("metadata", None)
        return EvaluatorAttachment(slug=str(slug), metadata=metadata)
    raise TypeError(f"Unsupported evaluator specification: {entry!r}")


def _normalize_evaluators(evaluators: Sequence[Any] | None) -> list[EvaluatorAttachment]:
    if not evaluators:
        return []
    if not isinstance(evaluators, Sequence) or isinstance(evaluators, (str, bytes)):
        evaluators = [evaluators]
    result: list[EvaluatorAttachment] = []
    for entry in evaluators:
        if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes, EvaluatorAttachment)):
            result.extend(_normalize_evaluators(entry))
        else:
            result.append(_normalize_evaluator_entry(entry))
    return result


def normalize_evaluator_specs(evaluators: Sequence[Any] | None) -> list[EvaluatorAttachment]:
    """Public helper to normalize evaluator specifications."""
    return _normalize_evaluators(evaluators)


@contextmanager
def with_evaluators(
    evaluators: Sequence[Any] | None,
    config: EvaluatorConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Generator[None, None, None]:
    """Propagate evaluator slugs, config, and metadata through the OpenTelemetry context.

    Args:
        evaluators: Evaluator specifications to propagate.
        config: Optional evaluator config to attach to spans.
        metadata: Optional evaluator metadata to attach to spans.
    """

    attachments = normalize_evaluator_specs(evaluators)
    if not attachments and not config and not metadata:
        yield
        return

    # Propagate evaluator slugs
    tokens = []
    if attachments:
        existing = otel_context.get_value(EVALUATOR_CONTEXT_KEY)
        combined: list[str] = []
        if isinstance(existing, (list, tuple)):
            combined.extend(str(slug) for slug in existing if str(slug).strip())

        for attachment in attachments:
            if attachment.slug not in combined:
                combined.append(attachment.slug)

        tokens.append(attach(set_value(EVALUATOR_CONTEXT_KEY, tuple(combined))))

    # Propagate evaluator config
    if config is not None:
        tokens.append(attach(set_value(EVALUATOR_CONFIG_CONTEXT_KEY, config)))

    # Propagate evaluator metadata
    if metadata is not None:
        # Merge with existing metadata
        existing_metadata = otel_context.get_value(EVALUATOR_METADATA_CONTEXT_KEY)
        merged_metadata = dict(existing_metadata) if isinstance(existing_metadata, Mapping) else {}
        merged_metadata.update(metadata)
        tokens.append(attach(set_value(EVALUATOR_METADATA_CONTEXT_KEY, merged_metadata)))

    try:
        yield
    finally:
        for token in reversed(tokens):
            detach(token)


def _attach_attributes(span: Span, attributes: dict[str, Any] | None) -> None:
    if not attributes:
        return
    for key, value in attributes.items():
        span.set_attribute(key, value)


def _serialize_attribute(value: Any) -> Any | None:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    try:
        return json.dumps(value)
    except Exception:
        return str(value)


def _set_serialized_attribute(span: Span, key: str, value: Any) -> None:
    serialized = _serialize_attribute(value)
    if serialized is not None:
        span.set_attribute(key, serialized)


def trace_content_enabled() -> bool:
    flag = os.getenv("TRACELOOP_TRACE_CONTENT")
    if flag is None:
        return True
    return flag.strip().lower() not in {"0", "false", "no", "off"}


def get_tracer(tracer_name: str = "basalt.observability") -> Tracer:
    """Get or create a tracer with the given name."""
    return trace.get_tracer(tracer_name)


class SpanHandle:
    """Helper around an OTEL span with convenience methods."""

    def __init__(
        self,
        span: Span,
        parent_span: Span | None = None,
        defaults: TraceContextConfig | None = None,
    ):
        self._span = span
        self._io_payload: dict[str, Any] = {"input": None, "output": None, "variables": None}
        self._parent_span = parent_span if parent_span and parent_span.get_span_context().is_valid else None
        self._evaluators: dict[str, EvaluatorAttachment] = {}
        self._evaluator_config: EvaluatorConfig | None = None
        self._evaluator_metadata: dict[str, Any] = {}
        self._hydrate_existing_evaluators()

        # Apply config from context if available
        context_config = otel_context.get_value(EVALUATOR_CONFIG_CONTEXT_KEY)
        if context_config and isinstance(context_config, EvaluatorConfig):
            self.set_evaluator_config(context_config)

        # Apply metadata from context if available
        context_metadata = otel_context.get_value(EVALUATOR_METADATA_CONTEXT_KEY)
        if context_metadata and isinstance(context_metadata, Mapping):
            self.set_evaluator_metadata(context_metadata)

        if defaults and defaults.evaluators:
            for attachment in normalize_evaluator_specs(defaults.evaluators):
                self._append_evaluator(attachment)


    def set_attribute(self, key: str, value: Any) -> None:
        """
        Sets an attribute on the current span.

        Args:
            key (str): The name of the attribute to set.
            value (Any): The value of the attribute.

        Returns:
            None
        """
        self._span.set_attribute(key, value)

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """
        Add a custom event to the current span.

        Args:
            name: Event name.
            attributes: Optional event attributes.
        """
        self._span.add_event(name, attributes=attributes)

    def set_status(self, status_code: StatusCode, description: str | None = None) -> None:
        """
        Sets the status of the current span.
        Args:
            status_code (StatusCode): The status code to set.
            description (str | None): An optional description for the status.
        Returns:
            None
        """
        self._span.set_status(Status(status_code, description))

    def record_exception(self, exc: BaseException) -> None:
        """
        Record an exception on the current span.
        Args:
            exc (BaseException): The exception to record.
        Returns:
            None
        """
        self._span.record_exception(exc)

    # ------------------------------------------------------------------
    # IO helpers
    # ------------------------------------------------------------------
    def set_input(self, payload: Any) -> None:
        """
        Sets the input payload for the current context manager.
        Stores the provided payload in the internal `_io_payload` dictionary under the "input" key.
        If trace content is enabled, serializes and attaches the input payload to the tracing span
        using the appropriate semantic convention.
        Args:
            payload (Any): The input data to be recorded and optionally traced.
        """

        self._io_payload["input"] = payload
        if trace_content_enabled():
            _set_serialized_attribute(self._span, semconv.BasaltSpan.INPUT, payload)

    def set_output(self, payload: Any) -> None:
        """
        Sets the output payload for the current context manager.
        Stores the provided payload in the internal I/O payload dictionary under the "output" key.
        If trace content is enabled, serializes and attaches the payload to the current span for observability.
        Args:
            payload (Any): The output data to be stored and optionally traced.
        """

        self._io_payload["output"] = payload
        if trace_content_enabled():
            _set_serialized_attribute(self._span, semconv.BasaltSpan.OUTPUT, payload)

    def set_variables(self, variables: Mapping[str, Any] | None) -> None:
        """
        Sets the variables payload for the current context manager.
        Stores the provided variables in the internal `_io_payload` dictionary under the "variables" key.
        If trace content is enabled, serializes and attaches the variables to the tracing span
        using the appropriate semantic convention.
        Args:
            variables (Mapping[str, Any] | None): The variables data to be recorded and optionally traced.
        """
        if variables is None:
            return
        if not isinstance(variables, Mapping):
            raise TypeError("Span variables must be provided as a mapping.")
        self._io_payload["variables"] = dict(variables)
        if trace_content_enabled():
            _set_serialized_attribute(self._span, semconv.BasaltSpan.VARIABLES, variables)
            if self._parent_span:
                _set_serialized_attribute(self._parent_span, semconv.BasaltSpan.VARIABLES, variables)

    def set_io(
        self,
        *,
        input_payload: Any | None = None,
        output_payload: Any | None = None,
        variables: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Sets the input, output, and variables payloads for the current context manager.
        """
        if input_payload is not None:
            self.set_input(input_payload)
        if output_payload is not None:
            self.set_output(output_payload)
        if variables is not None:
            self.set_variables(variables)

    def io_snapshot(self) -> dict[str, Any]:
        """Return a shallow copy of the tracked IO payload."""
        snapshot = dict(self._io_payload)
        if snapshot["variables"] is not None:
            snapshot["variables"] = dict(snapshot["variables"])
        return snapshot

    # ------------------------------------------------------------------
    # Evaluators
    # ------------------------------------------------------------------
    def set_evaluator_config(self, config: EvaluatorConfig | Mapping[str, Any]) -> None:
        """Attach span-scoped evaluator configuration.

        The configuration applies to all evaluators attached to this span.
        It is stored under the semantic key BasaltSpan.EVALUATORS_CONFIG as JSON.

        Args:
            config: Either an EvaluatorConfig instance or a mapping with config values.
        """
        if isinstance(config, EvaluatorConfig):
            self._evaluator_config = config
            _set_serialized_attribute(self._span, semconv.BasaltSpan.EVALUATORS_CONFIG, config.to_dict())
        elif isinstance(config, Mapping):
            config_dict = dict(config)
            self._evaluator_config = EvaluatorConfig(**config_dict)
            _set_serialized_attribute(self._span, semconv.BasaltSpan.EVALUATORS_CONFIG, config_dict)
        else:
            raise TypeError("Evaluator config must be an EvaluatorConfig or a mapping.")

    def set_evaluator_metadata(self, metadata: Mapping[str, Any]) -> None:
        """Set span-scoped evaluator metadata.

        The metadata applies to all evaluators attached to this span.
        Metadata is stored as separate span attributes under the evaluator prefix.

        Args:
            metadata: A mapping of metadata key-value pairs.
        """
        if not isinstance(metadata, Mapping):
            raise TypeError("Evaluator metadata must be a mapping.")
        self._evaluator_metadata.update(metadata)
        for key, value in metadata.items():
            attr_key = f"{semconv.BasaltSpan.EVALUATOR_PREFIX}.metadata.{key}"
            _set_serialized_attribute(self._span, attr_key, value)

    def add_evaluator(
        self,
        evaluator_slug: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Attach an evaluator slug to the span.

        Args:
            evaluator_slug: The evaluator slug to attach.
            metadata: Optional metadata specific to this evaluator attachment.
        """
        attachment = EvaluatorAttachment(slug=evaluator_slug, metadata=metadata)
        self._append_evaluator(attachment)

    def add_evaluators(self, *evaluators: Any) -> None:
        """Attach multiple evaluators to the span."""
        for attachment in normalize_evaluator_specs(evaluators):
            self._append_evaluator(attachment)

    def set_user(self, user_id: str, name: str | None = None) -> None:
        """
            Set the user identity for the span.
        """
        self._span.set_attribute(semconv.BasaltUser.ID, user_id)
        if name:
            self._span.set_attribute(semconv.BasaltUser.NAME, name)

    def set_organization(self, organization_id: str, name: str | None = None) -> None:
        """
        Set the organization identity for the span.
        """
        self._span.set_attribute(semconv.BasaltOrganization.ID, organization_id)
        if name:
            self._span.set_attribute(semconv.BasaltOrganization.NAME, name)

    def set_experiment(
        self,
        experiment_id: str,
        *,
        name: str | None = None,
        feature_slug: str | None = None,
    ) -> None:
        """
        Set the experiment identity for the span.
        """
        self._span.set_attribute(semconv.BasaltExperiment.ID, experiment_id)
        if name:
            self._span.set_attribute(semconv.BasaltExperiment.NAME, name)
        if feature_slug:
            self._span.set_attribute(semconv.BasaltExperiment.FEATURE_SLUG, feature_slug)

    def _append_evaluator(self, attachment: EvaluatorAttachment) -> None:
        """Attach an evaluator to the span, avoiding duplicates.

        If the attachment includes metadata, it will be merged into the span-level metadata.
        """
        existing = self._evaluators.get(attachment.slug)
        if existing and existing == attachment:
            return
        self._evaluators[attachment.slug] = attachment
        evaluator_list = list(self._evaluators.keys())
        self._span.set_attribute(semconv.BasaltSpan.EVALUATORS, evaluator_list)

        # Merge metadata from attachment into span-level metadata
        if attachment.metadata:
            self.set_evaluator_metadata(attachment.metadata)

    def _hydrate_existing_evaluators(self) -> None:
        """Populate evaluator cache from span attributes if present."""

        attributes = getattr(self._span, "attributes", None)
        if not isinstance(attributes, dict):
            return
        current = attributes.get(semconv.BasaltSpan.EVALUATORS)
        if not isinstance(current, (list, tuple)):
            return
        for slug in current:
            if isinstance(slug, str) and slug.strip():
                normalized = EvaluatorAttachment(slug=slug)
                self._evaluators.setdefault(normalized.slug, normalized)

    @property
    def span(self) -> Span:
        return self._span


class LLMSpanHandle(SpanHandle):
    """
    Span handle with helpers for LLM/GenAI calls.

    Follows OpenTelemetry GenAI semantic conventions.
    See: https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/
    """

    def set_model(self, model: str) -> None:
        """
        Set the model name for the request.
        Uses gen_ai.request.model attribute.
        """
        self.set_attribute(semconv.GenAI.REQUEST_MODEL, model)

    def set_response_model(self, model: str) -> None:
        """
        Set the model name that generated the response.
        Uses gen_ai.response.model attribute.
        """
        self.set_attribute(semconv.GenAI.RESPONSE_MODEL, model)

    def set_prompt(self, prompt: str) -> None:
        """
        Set the prompt/input text.
        Note: This uses gen_ai.input.messages for structured messages.
        For simple string prompts, wraps in a message structure.
        """
        super().set_input(prompt)
        if trace_content_enabled():
            # Store as structured message format per OpenTelemetry spec
            messages = [{"role": "user", "parts": [{"type": "text", "content": prompt}]}]
            self.set_attribute(semconv.GenAI.INPUT_MESSAGES, json.dumps(messages))

    def set_completion(self, completion: str) -> None:
        """
        Set the completion/output text.
        Note: This uses gen_ai.output.messages for structured messages.
        For simple string completions, wraps in a message structure.
        """
        super().set_output(completion)
        if trace_content_enabled():
            # Store as structured message format per OpenTelemetry spec
            messages = [
                {
                    "role": "assistant",
                    "parts": [{"type": "text", "content": completion}],
                    "finish_reason": "stop",
                }
            ]
            self.set_attribute(semconv.GenAI.OUTPUT_MESSAGES, json.dumps(messages))

    def set_tokens(self, *, input: int | None = None, output: int | None = None) -> None:
        """
        Set token usage counts.
        Uses gen_ai.usage.input_tokens and gen_ai.usage.output_tokens attributes.
        """
        if input is not None:
            self.set_attribute(semconv.GenAI.USAGE_INPUT_TOKENS, input)
        if output is not None:
            self.set_attribute(semconv.GenAI.USAGE_OUTPUT_TOKENS, output)

    def set_operation_name(self, operation: str) -> None:
        """
        Set the GenAI operation name (e.g., "chat", "text_completion").
        This is a required attribute per OpenTelemetry GenAI spec.
        """
        self.set_attribute(semconv.GenAI.OPERATION_NAME, operation)

    def set_provider(self, provider: str) -> None:
        """
        Set the GenAI provider name (e.g., "openai", "anthropic").
        This is a required attribute per OpenTelemetry GenAI spec.
        """
        self.set_attribute(semconv.GenAI.PROVIDER_NAME, provider)

    def set_response_id(self, response_id: str) -> None:
        """Set the unique response/completion ID."""
        self.set_attribute(semconv.GenAI.RESPONSE_ID, response_id)

    def set_finish_reasons(self, reasons: list[str]) -> None:
        """Set the finish reasons array."""
        self.set_attribute(semconv.GenAI.RESPONSE_FINISH_REASONS, reasons)

    def set_temperature(self, temperature: float) -> None:
        """Set the temperature parameter."""
        self.set_attribute(semconv.GenAI.REQUEST_TEMPERATURE, temperature)

    def set_top_p(self, top_p: float) -> None:
        """Set the top_p parameter."""
        self.set_attribute(semconv.GenAI.REQUEST_TOP_P, top_p)

    def set_top_k(self, top_k: float) -> None:
        """Set the top_k parameter."""
        self.set_attribute(semconv.GenAI.REQUEST_TOP_K, top_k)

    def set_max_tokens(self, max_tokens: int) -> None:
        """Set the max_tokens parameter."""
        self.set_attribute(semconv.GenAI.REQUEST_MAX_TOKENS, max_tokens)


class RetrievalSpanHandle(SpanHandle):
    """
    Span handle for vector DB/retrieval operations.

    Uses Basalt-specific semantic conventions for retrieval operations.
    """

    def set_query(self, query: str) -> None:
        """Set the query text for the retrieval operation."""
        super().set_input(query)
        self.set_attribute(semconv.BasaltRetrieval.QUERY, query)

    def set_results_count(self, count: int) -> None:
        """Set the number of results returned."""
        self.set_attribute(semconv.BasaltRetrieval.RESULTS_COUNT, count)

    def set_top_k(self, top_k: int) -> None:
        """Set the top-K parameter for retrieval."""
        self.set_attribute(semconv.BasaltRetrieval.TOP_K, top_k)


class FunctionSpanHandle(SpanHandle):
    """
    Span handle for compute/function execution spans.
    """

    def set_function_name(self, function_name: str) -> None:
        """Set the logical function name being executed."""
        self.set_attribute(semconv.BasaltFunction.NAME, function_name)

    def set_stage(self, stage: str) -> None:
        """Set the stage or phase associated with the execution."""
        self.set_attribute(semconv.BasaltFunction.STAGE, stage)

    def add_metric(self, key: str, value: Any) -> None:
        """Attach custom metric data to the function execution."""
        self.set_attribute(f"{semconv.BasaltFunction.METRIC_PREFIX}.{key}", value)


class ToolSpanHandle(SpanHandle):
    """
    Span handle for tool invocation spans.

    Uses Basalt-specific semantic conventions for tool operations.
    """

    def set_tool_name(self, name: str) -> None:
        """Set the name of the tool being invoked."""
        self.set_attribute(semconv.BasaltTool.NAME, name)

    def set_input(self, payload: Any) -> None:
        """Set the input payload for the tool."""
        super().set_input(payload)
        if trace_content_enabled():
            value = _serialize_attribute(payload)
            if value is not None:
                self.set_attribute(semconv.BasaltTool.INPUT, value)

    def set_output(self, payload: Any) -> None:
        """Set the output payload from the tool."""
        super().set_output(payload)
        if trace_content_enabled():
            value = _serialize_attribute(payload)
            if value is not None:
                self.set_attribute(semconv.BasaltTool.OUTPUT, value)


class EventSpanHandle(SpanHandle):
    """
    Span handle for custom application events.

    Uses Basalt-specific semantic conventions for event operations.
    """

    def set_event_type(self, event_type: str) -> None:
        """Set the type of custom event."""
        self.set_attribute(semconv.BasaltEvent.TYPE, event_type)

    def set_payload(self, payload: Any) -> None:
        """Set the event payload."""
        super().set_input(payload)
        if trace_content_enabled():
            value = _serialize_attribute(payload)
            if value is not None:
                self.set_attribute(semconv.BasaltEvent.PAYLOAD, value)


@contextmanager
def _with_span_handle(
    name: str,
    attributes: dict[str, Any] | None,
    tracer_name: str,
    handle_cls: type[SpanHandle],
    span_type: str | None = None,
    *,
    input_payload: Any | None = None,
    output_payload: Any | None = None,
    variables: Mapping[str, Any] | None = None,
    evaluators: Sequence[Any] | None = None,
    user: TraceIdentity | Mapping[str, Any] | None = None,
    organization: TraceIdentity | Mapping[str, Any] | None = None,
    ensure_output: bool = True,
) -> Generator[SpanHandle, None, None]:
    tracer = get_tracer(tracer_name)
    defaults = current_trace_defaults()

    parent_span = trace.get_current_span()
    if parent_span and not parent_span.get_span_context().is_valid:
        parent_span = None

    # Prepare context tokens for user/org propagation
    tokens = []
    if user is not None:
        from .trace_context import _coerce_identity
        user_identity = _coerce_identity(user)
        if user_identity:
            tokens.append(attach(set_value(USER_CONTEXT_KEY, user_identity)))

    if organization is not None:
        from .trace_context import _coerce_identity
        org_identity = _coerce_identity(organization)
        if org_identity:
            tokens.append(attach(set_value(ORGANIZATION_CONTEXT_KEY, org_identity)))

    try:
        with tracer.start_as_current_span(name) as span:
            _attach_attributes(span, attributes)
            if span_type:
                span.set_attribute(SPAN_TYPE_ATTRIBUTE, span_type)

            # Apply user/org from context (either explicit or inherited from parent)
            apply_user_from_context(span, user)
            apply_organization_from_context(span, organization)

            handle = handle_cls(span, parent_span, defaults)
            if input_payload is not None:
                handle.set_input(input_payload)
            if variables:
                handle.set_variables(variables)
            if evaluators:
                handle.add_evaluators(*evaluators)
            yield handle  # type: ignore[misc]
            if output_payload is not None:
                handle.set_output(output_payload)
            elif ensure_output and trace_content_enabled():
                io = handle.io_snapshot()
                if io.get("output") is None:
                    logger.debug("Span '%s' completed without an output payload.", name)
    finally:
        # Detach context tokens in reverse order
        for token in reversed(tokens):
            detach(token)


@contextmanager
def trace_span(
    name: str,
    *,
    input_payload: Any | None = None,
    output_payload: Any | None = None,
    variables: Mapping[str, Any] | None = None,
    evaluators: Sequence[Any] | None = None,
    user: TraceIdentity | Mapping[str, Any] | None = None,
    organization: TraceIdentity | Mapping[str, Any] | None = None,
    attributes: dict[str, Any] | None = None,
    tracer_name: str = "basalt.observability",
    span_type: str | None = "span",
) -> Generator[SpanHandle, None, None]:
    """
    Context manager for a generic span.
    Args:
        name: The name of the span.
        input_payload: Optional input payload to record.
        output_payload: Optional output payload to record.
        variables: Optional variables mapping to record.
        evaluators: Optional list of evaluator specifications to attach.
        user: Optional user identity for this span (propagates to children).
        organization: Optional organization identity for this span (propagates to children).
        attributes: Optional dictionary of attributes to set on the span.
        tracer_name: The name of the tracer to use.
        span_type: Optional type of the span (e.g., "generation", "retrieval").
    Yields:
        SpanHandle: A handle to interact with the span within the context.
    """
    if input_payload is None:
        logger.debug("trace_span '%s' initialized without an input payload.", name)
    with _with_span_handle(
        name,
        attributes,
        tracer_name,
        SpanHandle,
        span_type=span_type,
        input_payload=input_payload,
        output_payload=output_payload,
        variables=variables,
        evaluators=evaluators,
        user=user,
        organization=organization,
    ) as handle:
        yield handle


@contextmanager
def trace_generation(
    name: str,
    *,
    input_payload: Any | None = None,
    output_payload: Any | None = None,
    variables: Mapping[str, Any] | None = None,
    evaluators: Sequence[Any] | None = None,
    user: TraceIdentity | Mapping[str, Any] | None = None,
    organization: TraceIdentity | Mapping[str, Any] | None = None,
    attributes: dict[str, Any] | None = None,
    tracer_name: str = "basalt.observability.generation",
) -> Generator[LLMSpanHandle, None, None]:
    """Context manager for LLM/GenAI generation spans."""
    with _with_span_handle(
        name,
        attributes,
        tracer_name,
        LLMSpanHandle,
        span_type="generation",
        input_payload=input_payload,
        output_payload=output_payload,
        variables=variables,
        evaluators=evaluators,
        user=user,
        organization=organization,
    ) as handle:
        yield handle  # type: ignore[misc]


@contextmanager
def trace_retrieval(
    name: str,
    *,
    input_payload: Any | None = None,
    output_payload: Any | None = None,
    variables: Mapping[str, Any] | None = None,
    evaluators: Sequence[Any] | None = None,
    user: TraceIdentity | Mapping[str, Any] | None = None,
    organization: TraceIdentity | Mapping[str, Any] | None = None,
    attributes: dict[str, Any] | None = None,
    tracer_name: str = "basalt.observability.retrieval",
) -> Generator[RetrievalSpanHandle, None, None]:
    """Context manager for retrieval/vector DB spans."""
    with _with_span_handle(
        name,
        attributes,
        tracer_name,
        RetrievalSpanHandle,
        span_type="retrieval",
        input_payload=input_payload,
        output_payload=output_payload,
        variables=variables,
        evaluators=evaluators,
        user=user,
        organization=organization,
    ) as handle:
        yield handle  # type: ignore[misc]


@contextmanager
def trace_function(
    name: str,
    *,
    input_payload: Any | None = None,
    output_payload: Any | None = None,
    variables: Mapping[str, Any] | None = None,
    evaluators: Sequence[Any] | None = None,
    user: TraceIdentity | Mapping[str, Any] | None = None,
    organization: TraceIdentity | Mapping[str, Any] | None = None,
    attributes: dict[str, Any] | None = None,
    tracer_name: str = "basalt.observability.function",
) -> Generator[FunctionSpanHandle, None, None]:
    """Context manager for compute/function spans."""
    with _with_span_handle(
        name,
        attributes,
        tracer_name,
        FunctionSpanHandle,
        span_type="function",
        input_payload=input_payload,
        output_payload=output_payload,
        variables=variables,
        evaluators=evaluators,
        user=user,
        organization=organization,
    ) as handle:
        yield handle  # type: ignore[misc]


@contextmanager
def trace_tool(
    name: str,
    *,
    input_payload: Any | None = None,
    output_payload: Any | None = None,
    variables: Mapping[str, Any] | None = None,
    evaluators: Sequence[Any] | None = None,
    user: TraceIdentity | Mapping[str, Any] | None = None,
    organization: TraceIdentity | Mapping[str, Any] | None = None,
    attributes: dict[str, Any] | None = None,
    tracer_name: str = "basalt.observability.tool",
) -> Generator[ToolSpanHandle, None, None]:
    """Context manager for tool invocation spans."""
    with _with_span_handle(
        name,
        attributes,
        tracer_name,
        ToolSpanHandle,
        span_type="tool",
        input_payload=input_payload,
        output_payload=output_payload,
        variables=variables,
        evaluators=evaluators,
        user=user,
        organization=organization,
    ) as handle:
        yield handle  # type: ignore[misc]


@contextmanager
def trace_event(
    name: str,
    *,
    input_payload: Any | None = None,
    output_payload: Any | None = None,
    variables: Mapping[str, Any] | None = None,
    evaluators: Sequence[Any] | None = None,
    user: TraceIdentity | Mapping[str, Any] | None = None,
    organization: TraceIdentity | Mapping[str, Any] | None = None,
    attributes: dict[str, Any] | None = None,
    tracer_name: str = "basalt.observability.event",
) -> Generator[EventSpanHandle, None, None]:
    """Context manager for custom application event spans."""
    with _with_span_handle(
        name,
        attributes,
        tracer_name,
        EventSpanHandle,
        span_type="event",
        input_payload=input_payload,
        output_payload=output_payload,
        variables=variables,
        evaluators=evaluators,
        user=user,
        organization=organization,
    ) as handle:
        yield handle  # type: ignore[misc]


@contextmanager
def trace_llm_call(
    name: str,
    *,
    input_payload: Any | None = None,
    output_payload: Any | None = None,
    variables: Mapping[str, Any] | None = None,
    evaluators: Sequence[Any] | None = None,
    user: TraceIdentity | Mapping[str, Any] | None = None,
    organization: TraceIdentity | Mapping[str, Any] | None = None,
    attributes: dict[str, Any] | None = None,
    tracer_name: str = "basalt.observability.llm",
) -> Generator[LLMSpanHandle, None, None]:
    """
    Deprecated alias for ``trace_generation``.
    """
    warnings.warn(
        "trace_llm_call is deprecated; use trace_generation instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    with trace_generation(
        name,
        input_payload=input_payload,
        output_payload=output_payload,
        variables=variables,
        evaluators=evaluators,
        user=user,
        organization=organization,
        attributes=attributes,
        tracer_name=tracer_name,
    ) as handle:
        yield handle
