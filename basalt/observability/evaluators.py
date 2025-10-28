"""Evaluator management for sampling and attaching evaluators to spans."""

from __future__ import annotations

import random
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import Any

from opentelemetry import trace

from .context_managers import SpanHandle


@dataclass(frozen=True, slots=True)
class EvaluatorConfig:
    """Configuration for an evaluator with sampling support."""

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
        config = EvaluatorConfig(slug=slug, sample_rate=sample_rate, metadata=metadata)
        self.register(config)

    def unregister(self, slug: str) -> None:
        """Remove an evaluator configuration."""
        with self._lock:
            self._evaluators.pop(slug, None)

    def get_config(self, slug: str) -> EvaluatorConfig | None:
        """Retrieve an evaluator configuration by slug."""
        with self._lock:
            return self._evaluators.get(slug)

    def should_sample(self, slug: str) -> bool:
        """Determine if an evaluator should be attached based on sample rate."""
        with self._lock:
            config = self._evaluators.get(slug)
            if config is None:
                # If not registered, assume 100% sampling for backward compatibility
                return True
            return random.random() < config.sample_rate

    def attach_to_span(self, span_handle: SpanHandle, *evaluator_slugs: str) -> None:
        """Attach evaluators to a span, respecting sample rates."""
        for slug in evaluator_slugs:
            if slug and self.should_sample(slug):
                span_handle.add_evaluator(slug)

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
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Register an evaluator with optional sampling configuration.

    Args:
        slug: Unique identifier for the evaluator.
        sample_rate: Probability (0.0-1.0) that this evaluator will be attached to spans.
        metadata: Optional metadata associated with this evaluator.

    Example:
        >>> register_evaluator("hallucination-check", sample_rate=0.5)
        >>> register_evaluator("quality-eval", sample_rate=1.0)
    """
    _evaluator_manager.register_evaluator(slug, sample_rate, metadata)


def unregister_evaluator(slug: str) -> None:
    """
    Remove an evaluator configuration.

    Args:
        slug: The evaluator slug to unregister.
    """
    _evaluator_manager.unregister(slug)


@contextmanager
def attach_evaluator(
    *evaluator_slugs: str,
    span: SpanHandle | None = None,
) -> Generator[None, None, None]:
    """
    Context manager to attach evaluators to the current or specified span.

    Evaluators are attached respecting their configured sample rates. If an evaluator
    is not registered, it will be attached with 100% probability.

    Args:
        *evaluator_slugs: One or more evaluator slugs to attach.
        span: Optional span handle to attach to. If None, uses current span.

    Example:
        >>> with trace_llm_call("my.llm") as llm_span:
        ...     with attach_evaluator("hallucination-check", "quality-eval"):
        ...         llm_span.set_model("gpt-4")
        ...         llm_span.set_prompt("Tell me a joke")
        ...         result = call_llm()
        ...         llm_span.set_completion(result)

    Example with explicit span:
        >>> with trace_llm_call("my.llm") as llm_span:
        ...     with attach_evaluator("hallucination-check", span=llm_span):
        ...         result = call_llm()
    """
    target_span = span
    if target_span is None:
        # Get current OpenTelemetry span and wrap it
        otel_span = trace.get_current_span()
        if otel_span and otel_span.get_span_context().is_valid:
            # Wrap the span in a SpanHandle for evaluator attachment
            target_span = SpanHandle(otel_span)

    if target_span:
        _evaluator_manager.attach_to_span(target_span, *evaluator_slugs)

    yield


def attach_evaluators_to_span(span_handle: SpanHandle, *evaluator_slugs: str) -> None:
    """
    Directly attach evaluators to a span handle, respecting sample rates.

    Args:
        span_handle: The span handle to attach evaluators to.
        *evaluator_slugs: One or more evaluator slugs to attach.

    Example:
        >>> with trace_llm_call("my.llm") as llm_span:
        ...     attach_evaluators_to_span(llm_span, "hallucination-check", "quality-eval")
        ...     result = call_llm()
    """
    _evaluator_manager.attach_to_span(span_handle, *evaluator_slugs)


def attach_evaluators_to_current_span(*evaluator_slugs: str) -> None:
    """
    Attach evaluators to the current active span, respecting sample rates.

    This is a convenience function that finds the current OpenTelemetry span
    and attaches the specified evaluators to it.

    Args:
        *evaluator_slugs: One or more evaluator slugs to attach.

    Example:
        >>> # Inside an instrumented function or span context
        >>> attach_evaluators_to_current_span("hallucination-check", "quality-eval")
    """
    otel_span = trace.get_current_span()
    if otel_span and otel_span.get_span_context().is_valid:
        span_handle = SpanHandle(otel_span)
        _evaluator_manager.attach_to_span(span_handle, *evaluator_slugs)
