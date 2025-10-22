"""
Tracing decorators for the Basalt SDK.

This module provides decorators for adding OpenTelemetry tracing to functions and methods.
"""
import functools
import inspect
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

from opentelemetry.trace import Status, StatusCode

from .provider import get_tracer

P = ParamSpec("P")
R = TypeVar("R")


def trace_function(
    name: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator to trace a function with OpenTelemetry.

    This decorator automatically creates a span for the decorated function,
    capturing execution time and any exceptions that occur.

    Args:
        name: Optional custom name for the span. If not provided, uses the function's qualified name.
        attributes: Optional dictionary of attributes to attach to the span.

    Returns:
        The decorated function.

    Example:
        ```python
        from basalt.tracing.decorators import trace_function

        @trace_function()
        def my_function(x: int) -> int:
            return x * 2

        @trace_function(name="custom-operation", attributes={"user_id": "123"})
        async def async_function(data: str) -> str:
            return data.upper()
        ```

    Notes:
        - Works with both sync and async functions
        - Automatically captures exceptions and marks spans as errors
        - Uses the global tracer provider configured via setup_tracing()
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        # Determine span name
        span_name = name if name is not None else f"{func.__module__}.{func.__qualname__}"

        # Check if function is async
        is_async = inspect.iscoroutinefunction(func)

        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                # Get the tracer at runtime, not decoration time
                # This ensures we use the current global tracer provider
                tracer = get_tracer(func.__module__)
                with tracer.start_as_current_span(span_name) as span:
                    # Add custom attributes
                    if attributes:
                        for key, value in attributes.items():
                            span.set_attribute(key, value)

                    try:
                        result = await func(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        # Set error status - the exception is automatically recorded by the span context manager
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise

            return async_wrapper  # type: ignore
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                # Get the tracer at runtime, not decoration time
                # This ensures we use the current global tracer provider
                tracer = get_tracer(func.__module__)
                with tracer.start_as_current_span(span_name) as span:
                    # Add custom attributes
                    if attributes:
                        for key, value in attributes.items():
                            span.set_attribute(key, value)

                    try:
                        result = func(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        # Set error status - the exception is automatically recorded by the span context manager
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise

            return sync_wrapper  # type: ignore

    return decorator


# Alias for backward compatibility and convenience
trace = trace_function
