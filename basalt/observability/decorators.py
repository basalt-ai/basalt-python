"""Decorator utilities for Basalt observability."""

from __future__ import annotations

import enum
import functools
import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeAlias, TypeVar

from .context_managers import (
    with_evaluators,
)


class ObserveKind(str, enum.Enum):
    """Enumeration of span kinds for the observe decorator."""

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
    *,
    metadata: Mapping[str, Any] | Callable[..., Mapping[str, Any]] | None = None,
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
        metadata: Optional metadata for evaluators. Can be a static dict or a callable that
                 receives function arguments and returns a dict.

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

        def _resolve_metadata(args, kwargs):
            """Resolve metadata from callable or static value."""
            if metadata is None:
                return None
            if callable(metadata):
                try:
                    resolved = metadata(*args, **kwargs)
                    if resolved and isinstance(resolved, Mapping):
                        return resolved
                except Exception:
                    pass  # Silently skip if metadata resolution fails
                return None
            elif isinstance(metadata, Mapping):
                return metadata
            return None

        if is_async:

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not _should_attach():
                    return await func(*args, **kwargs)

                # Resolve metadata before entering context
                resolved_metadata = _resolve_metadata(args, kwargs)

                with with_evaluators(slug_list, config=None, metadata=resolved_metadata):
                    return await func(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not _should_attach():
                return func(*args, **kwargs)

            # Resolve metadata before entering context
            resolved_metadata = _resolve_metadata(args, kwargs)

            with with_evaluators(slug_list, config=None, metadata=resolved_metadata):
                return func(*args, **kwargs)

        return sync_wrapper  # type: ignore[return-value]

    return decorator
