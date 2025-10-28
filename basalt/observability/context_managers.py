"""Context manager utilities for manual telemetry spans."""

from __future__ import annotations

import json
import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode, Tracer

from . import semconv
from .trace_context import TraceContextConfig, apply_trace_defaults, current_trace_defaults

if TYPE_CHECKING:
    from .evaluators import EvaluatorManager

SPAN_TYPE_ATTRIBUTE = semconv.BasaltSpan.TYPE


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
        defaults: TraceContextConfig | None = None,
        evaluator_manager: EvaluatorManager | None = None,
    ):
        self._span = span
        self._evaluators: list[str] = []
        self._evaluator_manager = evaluator_manager
        if defaults and defaults.evaluators:
            self._apply_evaluators_with_sampling(defaults.evaluators)

    def _apply_evaluators_with_sampling(self, evaluator_slugs: list[str]) -> None:
        """Apply evaluators from defaults, respecting sample rates if manager is available."""
        if self._evaluator_manager:
            self._evaluator_manager.attach_to_span(self, *evaluator_slugs)
        else:
            # Fall back to adding all evaluators without sampling
            for slug in evaluator_slugs:
                self._append_evaluator(slug)

    def set_attribute(self, key: str, value: Any) -> None:
        self._span.set_attribute(key, value)

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self._span.add_event(name, attributes=attributes)

    def set_status(self, status_code: StatusCode, description: str | None = None) -> None:
        self._span.set_status(Status(status_code, description))

    def record_exception(self, exc: BaseException) -> None:
        self._span.record_exception(exc)

    def add_evaluator(self, evaluator_slug: str) -> None:
        """Attach an evaluator slug to the span."""
        self._append_evaluator(evaluator_slug)

    def set_user(self, user_id: str, name: str | None = None) -> None:
        self._span.set_attribute(semconv.BasaltUser.ID, user_id)
        if name:
            self._span.set_attribute(semconv.BasaltUser.NAME, name)

    def set_organization(self, organization_id: str, name: str | None = None) -> None:
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
        self._span.set_attribute(semconv.BasaltExperiment.ID, experiment_id)
        if name:
            self._span.set_attribute(semconv.BasaltExperiment.NAME, name)
        if feature_slug:
            self._span.set_attribute(semconv.BasaltExperiment.FEATURE_SLUG, feature_slug)

    def _append_evaluator(self, evaluator_slug: str) -> None:
        if not evaluator_slug or not isinstance(evaluator_slug, str):
            return
        if evaluator_slug not in self._evaluators:
            self._evaluators.append(evaluator_slug)
            self._span.set_attribute(semconv.BasaltTrace.EVALUATORS, list(self._evaluators))

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
        self.set_attribute(semconv.BasaltRetrieval.QUERY, query)

    def set_results_count(self, count: int) -> None:
        """Set the number of results returned."""
        self.set_attribute(semconv.BasaltRetrieval.RESULTS_COUNT, count)

    def set_top_k(self, top_k: int) -> None:
        """Set the top-K parameter for retrieval."""
        self.set_attribute(semconv.BasaltRetrieval.TOP_K, top_k)


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
        if trace_content_enabled():
            value = _serialize_attribute(payload)
            if value is not None:
                self.set_attribute(semconv.BasaltTool.INPUT, value)

    def set_output(self, payload: Any) -> None:
        """Set the output payload from the tool."""
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
) -> Generator[SpanHandle, None, None]:
    tracer = get_tracer(tracer_name)
    defaults = current_trace_defaults()

    # Get evaluator manager (lazy import to avoid circular dependency)
    evaluator_manager = None
    try:
        from .evaluators import get_evaluator_manager
        evaluator_manager = get_evaluator_manager()
    except ImportError:
        pass

    with tracer.start_as_current_span(name) as span:
        _attach_attributes(span, attributes)
        if span_type:
            span.set_attribute(SPAN_TYPE_ATTRIBUTE, span_type)
        apply_trace_defaults(span, defaults)
        handle = handle_cls(span, defaults, evaluator_manager)
        yield handle  # type: ignore[misc]


@contextmanager
def trace_span(
    name: str,
    attributes: dict[str, Any] | None = None,
    tracer_name: str = "basalt.observability",
    span_type: str | None = None,
) -> Generator[SpanHandle, None, None]:
    """Context manager for a generic span."""
    with _with_span_handle(
        name,
        attributes,
        tracer_name,
        SpanHandle,
        span_type=span_type,
    ) as handle:
        yield handle


@contextmanager
def trace_llm_call(
    name: str,
    attributes: dict[str, Any] | None = None,
    tracer_name: str = "basalt.observability.llm",
) -> Generator[LLMSpanHandle, None, None]:
    """Context manager for LLM spans."""
    with _with_span_handle(
        name,
        attributes,
        tracer_name,
        LLMSpanHandle,
        span_type="generation",
    ) as handle:
        yield handle  # type: ignore[misc]


@contextmanager
def trace_retrieval(
    name: str,
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
    ) as handle:
        yield handle  # type: ignore[misc]


@contextmanager
def trace_tool(
    name: str,
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
    ) as handle:
        yield handle  # type: ignore[misc]


@contextmanager
def trace_event(
    name: str,
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
    ) as handle:
        yield handle  # type: ignore[misc]
