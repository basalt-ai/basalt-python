"""
Evaluator management for sampling and attaching evaluators to spans.

This module provides functionality to register evaluators and attach them to traces
with optional sampling and attribute targeting.



### Working with Automatic Instrumentation:

**IMPORTANT**: Evaluators require manual span creation to work properly. They will NOT
be attached to spans created by automatic instrumentation (e.g., OpenTelemetry's
automatic LLM instrumentation) because the evaluator attachment happens in your
application code, not in the instrumented library.

To use evaluators with automatically instrumented LLM calls, you must wrap them with
manual tracing:

```python
@evaluator("my-eval")
@trace_generation(name="my.llm")
def my_llm_call():
    return openai.chat.completions.create(...)  # Auto-instrumented call
```

Without the manual `@trace_generation` wrapper, the evaluator will not be attached to the
span created by automatic instrumentation.


"""

from __future__ import annotations

import random
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import Any

from opentelemetry import trace

from .context_managers import SpanHandle, normalize_evaluator_specs


@dataclass(frozen=True, slots=True)
class EvaluatorConfig:
    """
    Configuration for an evaluator with sampling support.

    Attributes:
        slug: Unique identifier for the evaluator.
        sample_rate: Probability (0.0-1.0) that this evaluator will be attached to spans.
        metadata: Optional metadata associated with this evaluator.

    """

    slug: str
    sample_rate: float = 1.0
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.slug, str) or not self.slug:
            raise ValueError("Evaluator slug must be a non-empty string")
        if not 0.0 <= self.sample_rate <= 1.0:
            raise ValueError(f"Sample rate must be between 0.0 and 1.0, got {self.sample_rate}")


class EvaluatorManager:
    """Manages evaluator registration and sampling decisions."""

    def __init__(self) -> None:
        self._evaluators: dict[str, EvaluatorConfig] = {}
        self._lock = RLock()

    def register(self, config: EvaluatorConfig) -> None:
        """Register an evaluator configuration."""
        with self._lock:
            self._evaluators[config.slug] = config

    def register_evaluator(
        self,
        slug: str,
        sample_rate: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register an evaluator with the given slug and sample rate."""
        config = EvaluatorConfig(
            slug=slug,
            sample_rate=sample_rate,
            metadata=metadata,
        )
        self.register(config)

    def unregister(self, slug: str) -> None:
        """Remove an evaluator configuration."""
        with self._lock:
            self._evaluators.pop(slug, None)

    def get_config(self, slug: str) -> EvaluatorConfig | None:
        """Retrieve an evaluator configuration by slug."""
        with self._lock:
            return self._evaluators.get(slug)

    def attach_to_span(self, span_handle: SpanHandle, *evaluators: Any) -> None:
        """Attach evaluators to a span, respecting sample rates."""
        attachments = normalize_evaluator_specs(evaluators)
        for attachment in attachments:
            slug = attachment.slug
            config = self.get_config(slug)
            effective_rate = attachment.sample_rate
            if effective_rate is None and config is not None:
                effective_rate = config.sample_rate
            if effective_rate is None:
                effective_rate = 1.0
            if effective_rate < 1.0 and random.random() >= effective_rate:
                continue
            span_handle.add_evaluator(
                slug,
                sample_rate=effective_rate,
            )

    def list_evaluators(self) -> list[str]:
        """Return a list of all registered evaluator slugs."""
        with self._lock:
            return list(self._evaluators.keys())


# Global evaluator manager instance
_evaluator_manager = EvaluatorManager()


def get_evaluator_manager() -> EvaluatorManager:
    """Get the global evaluator manager instance."""
    return _evaluator_manager


def register_evaluator(
    slug: str,
    sample_rate: float = 1.0,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Register an evaluator with optional sampling configuration.

    Args:
        slug: Unique identifier for the evaluator.
        sample_rate: Probability (0.0-1.0) that this evaluator will be attached to spans.
                    Default is 1.0 (100%). Use lower values to reduce evaluation costs.
        metadata: Optional metadata associated with the evaluator.


    Example - Basic registration:
        >>> register_evaluator("hallucination-check", sample_rate=0.5)

    Example - Focus on completion only:
        >>> register_evaluator("quality-eval", sample_rate=1.0)

    Example - Focus on workflow attributes:
        >>> register_evaluator("answer-quality", sample_rate=0.75)

    Note:
        Evaluators require manual span creation (e.g., @trace_generation, trace_generation) to work.
        They will NOT be attached to spans created by automatic instrumentation unless
        you wrap the instrumented call with manual tracing. See module docstring for details.
    """
    _evaluator_manager.register_evaluator(
        slug,
        sample_rate=sample_rate,
        metadata=metadata,
    )


def unregister_evaluator(slug: str) -> None:
    """
    Remove an evaluator configuration.

    Args:
        slug: The evaluator slug to unregister.
    """
    _evaluator_manager.unregister(slug)


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
    target_span = span
    if target_span is None:
        # Get current OpenTelemetry span and wrap it
        otel_span = trace.get_current_span()
        if otel_span and otel_span.get_span_context().is_valid:
            # Wrap the span in a SpanHandle for evaluator attachment
            target_span = SpanHandle(otel_span)

    if target_span:
        _evaluator_manager.attach_to_span(target_span, *evaluators)

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
    _evaluator_manager.attach_to_span(span_handle, *evaluators)


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
        _evaluator_manager.attach_to_span(span_handle, *evaluators)
