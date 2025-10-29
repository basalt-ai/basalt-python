"""Helpers to trace Basalt API client requests without library auto-instrumentation."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from .context_managers import trace_span
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
    variables = dict(span_data.extra_attributes) if span_data.extra_attributes else None
    with trace_span(
        span_data.span_name(),
        input_payload=input_payload,
        variables=variables,
        attributes=span_data.start_attributes(),
    ) as span:
        try:
            result = await request_callable()
        except Exception as exc:  # pragma: no cover - passthrough
            span.set_output({"error": str(exc)})
            span_data.finalize(
                span,
                duration_s=time.perf_counter() - start,
                status_code=None,
                error=exc,
            )
            raise

        status_code = getattr(result, "status_code", None)
        span.set_output({"status_code": status_code})
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
    variables = dict(span_data.extra_attributes) if span_data.extra_attributes else None
    with trace_span(
        span_data.span_name(),
        input_payload=input_payload,
        variables=variables,
        attributes=span_data.start_attributes(),
    ) as span:
        try:
            result = request_callable()
        except Exception as exc:  # pragma: no cover - passthrough
            span.set_output({"error": str(exc)})
            span_data.finalize(
                span,
                duration_s=time.perf_counter() - start,
                status_code=None,
                error=exc,
            )
            raise

        status_code = getattr(result, "status_code", None)
        span.set_output({"status_code": status_code})
        span_data.finalize(
            span,
            duration_s=time.perf_counter() - start,
            status_code=status_code,
            error=None,
        )
        return result
