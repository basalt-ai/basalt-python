"""
Simplified evaluator utilities: attach evaluator slugs to spans with optional
span-scoped configuration.

In this simplified model:
- A span carries a list of evaluator slugs under basalt.span.evaluators
- Any evaluator-related configuration is span-scoped, not per-evaluator. You can attach
    configuration using basalt.span.evaluators.config (JSON-serialized) or attributes
    under the basalt.span.evaluator.* prefix.

Evaluators still require manual span creation (e.g., trace_generation/trace_span) for
attachment within your application code.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace

from .context_managers import SpanHandle, normalize_evaluator_specs, with_evaluators


def _flatten_evaluator_specs(*evaluators: Any) -> list[str]:
    """Normalize evaluator specifications into a flat list of slugs."""
    slugs: list[str] = []
    for attachment in normalize_evaluator_specs(evaluators):
        if attachment and isinstance(attachment.slug, str):
            slugs.append(attachment.slug)
    return slugs


@contextmanager
def attach_evaluator(
    *evaluators: Any,
    span: SpanHandle | None = None,
) -> Generator[None, None, None]:
    """
    Context manager to attach evaluators to the current or specified span.

    Evaluators are attached respecting their configured sample rates. If an evaluator is not registered,
    it will be attached with 100% probability.

    Args:
        *evaluators: One or more evaluator specifications to attach.
        span: Optional span handle to attach to. If None, uses current span.

    Example - Basic usage:
        >>> with trace_generation("my.llm") as llm_span:
        ...     with attach_evaluator("hallucination-check", "quality-eval"):
        ...         llm_span.set_model("gpt-4")
        ...         llm_span.set_prompt("Tell me a joke")
        ...         result = call_llm()
        ...         llm_span.set_completion(result)

    Example - With explicit span:
        >>> with trace_generation("my.llm") as llm_span:
        ...     with attach_evaluator("hallucination-check", span=llm_span):
        ...         result = call_llm()

    Note:
        This context manager works with manual span creation only. For automatic
        instrumentation, wrap your LLM call with manual tracing first.
        See module docstring for details.
    """
    slugs = _flatten_evaluator_specs(*evaluators)
    target_span = span
    if target_span is None:
        otel_span = trace.get_current_span()
        if otel_span and otel_span.get_span_context().is_valid:
            target_span = SpanHandle(otel_span)

    with with_evaluators(slugs):
        if target_span:
            for slug in slugs:
                target_span.add_evaluator(slug)
        yield


def attach_evaluators_to_span(span_handle: SpanHandle, *evaluators: Any) -> None:
    """
    Directly attach evaluators to a span handle, respecting sample rates.

    Args:
        span_handle: The span handle to attach evaluators to.
        *evaluators: One or more evaluator specifications to attach.

    Example:
        >>> with trace_generation("my.llm") as llm_span:
        ...     attach_evaluators_to_span(llm_span, "hallucination-check", "quality-eval")
        ...     result = call_llm()
    """
    for slug in _flatten_evaluator_specs(*evaluators):
        span_handle.add_evaluator(slug)


def attach_evaluators_to_current_span(*evaluators: Any) -> None:
    """
    Attach evaluators to the current active span, respecting sample rates.

    This is a convenience function that finds the current OpenTelemetry span
    and attaches the specified evaluators to it.
    Useful when you want to add evaluators outside of a context manager or decorator.
    It's working with automatic instrumentation as long as there is an active span.

    Args:
        *evaluators: One or more evaluator specifications to attach.

    Example:
        >>> # Inside an instrumented function or span context
        >>> attach_evaluators_to_current_span("hallucination-check", "quality-eval")
    """
    otel_span = trace.get_current_span()
    if otel_span and otel_span.get_span_context().is_valid:
        span_handle = SpanHandle(otel_span)
        for slug in _flatten_evaluator_specs(*evaluators):
            span_handle.add_evaluator(slug)
