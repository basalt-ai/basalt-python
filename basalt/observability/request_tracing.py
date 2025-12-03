"""Helpers to trace Basalt API client requests without library auto-instrumentation."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from .api import observe
from .spans import BasaltRequestSpan

T = TypeVar("T")


async def trace_async_request(
    span_data: BasaltRequestSpan,
    request_callable: Callable[[], Awaitable[T]],
) -> T:
    """
    Trace an asynchronous HTTP request.

    Args:
        span_data: Structured span description shared across SDK clients.
        request_callable: Awaitable factory performing the HTTP request.

    Returns:
        Result of ``request_callable``.
    """
    start = time.perf_counter()
    input_payload = {
        "method": span_data.method,
        "url": span_data.url,
    }

    with observe(name=span_data.span_name(), metadata=span_data.start_attributes()) as span:
        observe.set_input(input_payload)
        if span_data.variables:
            span.set_io(variables=span_data.variables)

        try:
            result = await request_callable()
        except Exception as exc:  # pragma: no cover - passthrough
            # If the exception carries an HTTP status_code (BasaltAPIError), include it
            status_code = getattr(exc, "status_code", None)
            observe.set_output({"error": str(exc), "status_code": status_code})
            span_data.finalize(
                span,
                duration_s=time.perf_counter() - start,
                status_code=status_code,
                error=exc,
            )
            raise

        # Type-safe output formatting for PromptRequestSpan
        from basalt.prompts.client import PromptRequestSpan
        if isinstance(span_data, PromptRequestSpan):
            output = span_data.format_output(result)
        else:
            status_code = getattr(result, "status_code", None)
            output = {"status_code": status_code}

        observe.set_output(output)
        status_code = getattr(result, "status_code", None)
        span_data.finalize(
            span,
            duration_s=time.perf_counter() - start,
            status_code=status_code,
            error=None,
        )
        return result


def trace_sync_request(
    span_data: BasaltRequestSpan,
    request_callable: Callable[[], T],
) -> T:
    """
    Trace a synchronous HTTP request.

    Args:
        span_data: Structured span description shared across SDK clients.
        request_callable: Callable performing the HTTP request.

    Returns:
        Result of ``request_callable``.
    """
    start = time.perf_counter()
    input_payload = {
        "method": span_data.method,
        "url": span_data.url,
    }

    with observe(name=span_data.span_name(), metadata=span_data.start_attributes()) as span:
        observe.set_input(input_payload)
        if span_data.variables:
            span.set_io(variables=span_data.variables)

        try:
            result = request_callable()
        except Exception as exc:  # pragma: no cover - passthrough
            # If the exception carries an HTTP status_code (BasaltAPIError), include it
            status_code = getattr(exc, "status_code", None)
            observe.set_output({"error": str(exc), "status_code": status_code})
            span_data.finalize(
                span,
                duration_s=time.perf_counter() - start,
                status_code=status_code,
                error=exc,
            )
            raise

        # Type-safe output formatting for PromptRequestSpan
        from basalt.prompts.client import PromptRequestSpan
        if isinstance(span_data, PromptRequestSpan):
            output = span_data.format_output(result)
        else:
            status_code = getattr(result, "status_code", None)
            output = {"status_code": status_code}

        observe.set_output(output)
        status_code = getattr(result, "status_code", None)
        span_data.finalize(
            span,
            duration_s=time.perf_counter() - start,
            status_code=status_code,
            error=None,
        )
        return result
