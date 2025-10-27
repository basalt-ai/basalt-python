"""Context manager utilities for manual telemetry spans."""

from __future__ import annotations

import json
import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode, Tracer

from .trace_context import TraceContextConfig, apply_trace_defaults, current_trace_defaults

SPAN_TYPE_ATTRIBUTE = "basalt.span.type"


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

    def __init__(self, span: Span, defaults: TraceContextConfig | None = None):
        self._span = span
        self._evaluators: list[str] = []
        if defaults and defaults.evaluators:
            for slug in defaults.evaluators:
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
        self._span.set_attribute("basalt.user.id", user_id)
        if name:
            self._span.set_attribute("basalt.user.name", name)

    def set_organization(self, organization_id: str, name: str | None = None) -> None:
        self._span.set_attribute("basalt.organization.id", organization_id)
        if name:
            self._span.set_attribute("basalt.organization.name", name)

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

    def _append_evaluator(self, evaluator_slug: str) -> None:
        if not evaluator_slug or not isinstance(evaluator_slug, str):
            return
        if evaluator_slug not in self._evaluators:
            self._evaluators.append(evaluator_slug)
            self._span.set_attribute("basalt.trace.evaluators", list(self._evaluators))

    @property
    def span(self) -> Span:
        return self._span


class LLMSpanHandle(SpanHandle):
    """Span handle with helpers suited for LLM calls."""

    def set_model(self, model: str) -> None:
        self.set_attribute("llm.model", model)

    def set_prompt(self, prompt: str) -> None:
        if trace_content_enabled():
            self.set_attribute("llm.prompt", prompt)

    def set_completion(self, completion: str) -> None:
        if trace_content_enabled():
            self.set_attribute("llm.completion", completion)

    def set_tokens(self, *, input: int | None = None, output: int | None = None) -> None:
        if input is not None:
            self.set_attribute("llm.tokens.input", input)
        if output is not None:
            self.set_attribute("llm.tokens.output", output)


class RetrievalSpanHandle(SpanHandle):
    """Span handle tailored to vector DB/retrieval events."""

    def set_query(self, query: str) -> None:
        self.set_attribute("retrieval.query", query)

    def set_results_count(self, count: int) -> None:
        self.set_attribute("retrieval.results.count", count)

    def set_top_k(self, top_k: int) -> None:
        self.set_attribute("retrieval.top_k", top_k)


class ToolSpanHandle(SpanHandle):
    """Span handle for tool invocation spans."""

    def set_tool_name(self, name: str) -> None:
        self.set_attribute("basalt.tool.name", name)

    def set_input(self, payload: Any) -> None:
        if trace_content_enabled():
            value = _serialize_attribute(payload)
            if value is not None:
                self.set_attribute("basalt.tool.input", value)

    def set_output(self, payload: Any) -> None:
        if trace_content_enabled():
            value = _serialize_attribute(payload)
            if value is not None:
                self.set_attribute("basalt.tool.output", value)


class EventSpanHandle(SpanHandle):
    """Span handle for emitting custom application events."""

    def set_event_type(self, event_type: str) -> None:
        self.set_attribute("basalt.event.type", event_type)

    def set_payload(self, payload: Any) -> None:
        if trace_content_enabled():
            value = _serialize_attribute(payload)
            if value is not None:
                self.set_attribute("basalt.event.payload", value)


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
    with tracer.start_as_current_span(name) as span:
        _attach_attributes(span, attributes)
        if span_type:
            span.set_attribute(SPAN_TYPE_ATTRIBUTE, span_type)
        apply_trace_defaults(span, defaults)
        handle = handle_cls(span, defaults)
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
