"""Context manager utilities for manual telemetry spans."""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode, Tracer


def _attach_attributes(span: Span, attributes: dict[str, Any] | None) -> None:
    if not attributes:
        return
    for key, value in attributes.items():
        span.set_attribute(key, value)


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

    def __init__(self, span: Span):
        self._span = span

    def set_attribute(self, key: str, value: Any) -> None:
        self._span.set_attribute(key, value)

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self._span.add_event(name, attributes=attributes)

    def set_status(self, status_code: StatusCode, description: str | None = None) -> None:
        self._span.set_status(Status(status_code, description))

    def record_exception(self, exc: BaseException) -> None:
        self._span.record_exception(exc)

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


@contextmanager
def trace_span(
    name: str,
    attributes: dict[str, Any] | None = None,
    tracer_name: str = "basalt.observability",
) -> Generator[SpanHandle, None, None]:
    """Context manager for a generic span."""
    tracer = get_tracer(tracer_name)
    with tracer.start_as_current_span(name) as span:
        _attach_attributes(span, attributes)
        yield SpanHandle(span)


@contextmanager
def trace_llm_call(
    name: str,
    attributes: dict[str, Any] | None = None,
    tracer_name: str = "basalt.observability.llm",
) -> Generator[LLMSpanHandle, None, None]:
    """Context manager for LLM spans."""
    tracer = get_tracer(tracer_name)
    with tracer.start_as_current_span(name) as span:
        _attach_attributes(span, attributes)
        yield LLMSpanHandle(span)


@contextmanager
def trace_retrieval(
    name: str,
    attributes: dict[str, Any] | None = None,
    tracer_name: str = "basalt.observability.retrieval",
) -> Generator[RetrievalSpanHandle, None, None]:
    """Context manager for retrieval/vector DB spans."""
    tracer = get_tracer(tracer_name)
    with tracer.start_as_current_span(name) as span:
        _attach_attributes(span, attributes)
        yield RetrievalSpanHandle(span)
