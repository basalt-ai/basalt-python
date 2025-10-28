"""
Evaluator management for sampling and attaching evaluators to spans.

This module provides functionality to register evaluators and attach them to traces
with optional sampling and attribute targeting.

## Understanding the `to_evaluate` Field

The `to_evaluate` field allows you to specify which attributes in a span the evaluator
should focus on when performing evaluation. This is particularly useful when you want
to evaluate specific parts of your LLM workflow or response.

### Default Behavior (when `to_evaluate` is None):

If no `to_evaluate` attributes are specified, the evaluator will use the following
fallback strategy:
1. First, try to use `gen_ai.output.messages` (the standard OpenTelemetry GenAI
   experimental attribute for structured output)
2. If that's not available, fall back to the completion content from the response

This default behavior ensures evaluators work with standard OpenTelemetry instrumented
LLM calls without requiring additional configuration.

### Custom Attribute Targeting:

You can specify custom attributes to focus the evaluation on specific parts of your trace:

```python
# Focus on just the completion
register_evaluator("quality", to_evaluate=["completion"])

# Focus on both prompt and completion
register_evaluator("hallucination", to_evaluate=["prompt", "completion"])

# Focus on workflow-specific attributes
register_evaluator("answer-eval", to_evaluate=["workflow.final_answer"])

# Focus on multiple workflow attributes
register_evaluator("multi-eval", to_evaluate=["context", "query", "answer"])
```

### Working with Automatic Instrumentation:

**IMPORTANT**: Evaluators require manual span creation to work properly. They will NOT
be attached to spans created by automatic instrumentation (e.g., OpenTelemetry's
automatic LLM instrumentation) because the evaluator attachment happens in your
application code, not in the instrumented library.

To use evaluators with automatically instrumented LLM calls, you must wrap them with
manual tracing:

```python
@evaluator("my-eval")
@trace_llm(name="my.llm")
def my_llm_call():
    return openai.chat.completions.create(...)  # Auto-instrumented call
```

Without the manual `@trace_llm` wrapper, the evaluator will not be attached to the
span created by automatic instrumentation.

### Backend Behavior:

When the backend processes a span with evaluators:
1. It checks if `basalt.trace.evaluator.{slug}.to_evaluate` exists
2. If it exists, it prioritizes those attributes for evaluation
3. If it doesn't exist or is None, it uses the default strategy:
   - Try `gen_ai.output.messages` (OpenTelemetry GenAI semantic convention)
   - Fall back to completion content if available

This allows the backend to focus on the most relevant data for each evaluator,
reducing token usage and improving evaluation accuracy.
"""

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
    """
    Configuration for an evaluator with sampling support.

    Attributes:
        slug: Unique identifier for the evaluator.
        sample_rate: Probability (0.0-1.0) that this evaluator will be attached to spans.
        metadata: Optional metadata associated with this evaluator.
        to_evaluate: Optional list of attribute names to focus evaluation on.
                    If None, the backend will use gen_ai.output.messages (OpenTelemetry GenAI
                    semantic convention) and fall back to completion content if unavailable.
                    See module docstring for detailed behavior.
    """

    slug: str
    sample_rate: float = 1.0
    metadata: dict[str, Any] | None = None
    to_evaluate: list[str] | None = None

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
        to_evaluate: list[str] | None = None,
    ) -> None:
        """Register an evaluator with the given slug and sample rate."""
        config = EvaluatorConfig(
            slug=slug,
            sample_rate=sample_rate,
            metadata=metadata,
            to_evaluate=to_evaluate,
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
                config = self.get_config(slug)
                to_evaluate = config.to_evaluate if config else None
                span_handle.add_evaluator(slug, to_evaluate=to_evaluate)

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
    to_evaluate: list[str] | None = None,
) -> None:
    """
    Register an evaluator with optional sampling configuration.

    Args:
        slug: Unique identifier for the evaluator.
        sample_rate: Probability (0.0-1.0) that this evaluator will be attached to spans.
                    Default is 1.0 (100%). Use lower values to reduce evaluation costs.
        metadata: Optional metadata associated with this evaluator.
        to_evaluate: List of attribute names to focus evaluation on.
                    If None (default), the backend will use gen_ai.output.messages
                    (OpenTelemetry GenAI semantic convention) and fall back to completion
                    content if unavailable. Specify attribute names to focus on specific
                    parts of your trace (e.g., ["completion"], ["workflow.final_answer"]).
                    See module docstring for detailed behavior.

    Example - Basic registration:
        >>> register_evaluator("hallucination-check", sample_rate=0.5)

    Example - Focus on completion only:
        >>> register_evaluator("quality-eval", sample_rate=1.0, to_evaluate=["completion"])

    Example - Focus on workflow attributes:
        >>> register_evaluator(
        ...     "answer-quality",
        ...     to_evaluate=["workflow.final_answer", "workflow.confidence"]
        ... )

    Note:
        Evaluators require manual span creation (e.g., @trace_llm, trace_llm_call) to work.
        They will NOT be attached to spans created by automatic instrumentation unless
        you wrap the instrumented call with manual tracing. See module docstring for details.
    """
    _evaluator_manager.register_evaluator(slug, sample_rate, metadata, to_evaluate)


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

    Evaluators are attached respecting their configured sample rates and to_evaluate
    settings. If an evaluator is not registered, it will be attached with 100%
    probability and default to_evaluate behavior (gen_ai.output.messages with
    completion fallback).

    Args:
        *evaluator_slugs: One or more evaluator slugs to attach.
        span: Optional span handle to attach to. If None, uses current span.

    Example - Basic usage:
        >>> with trace_llm_call("my.llm") as llm_span:
        ...     with attach_evaluator("hallucination-check", "quality-eval"):
        ...         llm_span.set_model("gpt-4")
        ...         llm_span.set_prompt("Tell me a joke")
        ...         result = call_llm()
        ...         llm_span.set_completion(result)

    Example - With explicit span:
        >>> with trace_llm_call("my.llm") as llm_span:
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
