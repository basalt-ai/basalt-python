"""Decorator utilities for Basalt observability."""

from __future__ import annotations

import enum
import functools
import inspect
from collections.abc import Callable, Sequence
from typing import Any, TypeAlias, TypeVar

from .context_managers import (
    with_evaluators,
)


class ObserveKind(str, enum.Enum):
    """Enumeration of span kinds for the observe decorator."""

    ROOT = "basalt_trace"
    SPAN = "span"
    GENERATION = "generation"
    RETRIEVAL = "retrieval"
    FUNCTION = "function"
    TOOL = "tool"
    EVENT = "event"


F = TypeVar("F", bound=Callable[..., Any])

# Type alias for span attributes: static dict, dynamic callable, or None
AttributeSpec: TypeAlias = dict[str, Any] | Callable[..., dict[str, Any]] | None


def evaluate(
    slugs: str | Sequence[str],
) -> Callable[[F], F]:
    """
    Decorator that propagates evaluator slugs through OpenTelemetry context.

    The decorator uses :func:`with_evaluators` to push evaluator identifiers into
    the active context. Any span created while the decorated function executes—
    whether by automatic instrumentation or manual tracing—receives the slugs via
    :class:`BasaltCallEvaluatorProcessor`. Manual spans also receive the slugs
    immediately through :class:`~basalt.observability.context_managers.SpanHandle`.

    Args:
        slugs: One or more evaluator slugs to attach.


    Example - Basic usage:
        >>> @evaluate("joke-quality")
        ... @observe(kind="generation", name="gemini.summarize_joke")
        ... def summarize_joke_with_gemini(joke: str) -> str:
        ...     return call_llm(joke)
    """

    if isinstance(slugs, str):
        slug_list = [slugs.strip()]
    elif isinstance(slugs, Sequence):
        slug_list = [str(slug).strip() for slug in slugs if str(slug).strip()]
    else:
        raise TypeError("Evaluator slugs must be provided as a string or sequence of strings.")

    slug_list = list(dict.fromkeys(slug_list))
    if not slug_list:
        raise ValueError("At least one evaluator slug must be provided.")

    def decorator(func: F) -> F:
        is_async = inspect.iscoroutinefunction(func)

        def _should_attach() -> bool:
            return True

        if is_async:

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not _should_attach():
                    return await func(*args, **kwargs)

                with with_evaluators(slug_list):
                    return await func(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not _should_attach():
                return func(*args, **kwargs)

            # Resolve metadata before entering context

            with with_evaluators(slug_list):
                return func(*args, **kwargs)

        return sync_wrapper  # type: ignore[return-value]

    return decorator
