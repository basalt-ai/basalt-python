"""Decorator utilities for Basalt observability."""

from __future__ import annotations

import functools
import inspect
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeAlias, TypeVar

from opentelemetry.trace import StatusCode

from .context_managers import (
    LLMSpanHandle,
    trace_content_enabled,
    trace_llm_call,
    trace_span,
)

F = TypeVar("F", bound=Callable[..., Any])
AsyncFunc = TypeVar("AsyncFunc", bound=Callable[..., Awaitable[Any]])

# Type alias for span attributes: static dict, dynamic callable, or None
AttributeSpec: TypeAlias = dict[str, Any] | Callable[..., dict[str, Any]] | None


def _resolve_attributes(
    attributes: AttributeSpec,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any] | None:
    if attributes is None:
        return None
    if callable(attributes):
        try:
            return attributes(*args, **kwargs)
        except Exception:
            return None
    return attributes


def _bind_args(func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]):
    try:
        signature = inspect.signature(func)
        return signature.bind_partial(*args, **kwargs)
    except Exception:
        return None


def _wrap_with_span(
    span_factory: Callable[..., Any],
    span_name: str,
    attributes: AttributeSpec,
    func: Callable[..., Any],
    apply_pre: Callable[[Any, Any], None] | None = None,
    apply_post: Callable[[Any, Any], None] | None = None,
) -> Callable[..., Any]:
    """
    Higher-order utility to wrap a function with span instrumentation.

    Parameters:
        span_factory: Context manager factory (e.g., trace_span, trace_llm_call)
        span_name: Name for the span
        attributes: Static attributes or callable to compute them
        func: Function to wrap
        apply_pre: Optional callback(span, bound_args) before execution
        apply_post: Optional callback(span, result) after execution

    Returns:
        Wrapped function (sync or async based on func)
    """
    is_async = inspect.iscoroutinefunction(func)

    if is_async:

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            computed_attrs = _resolve_attributes(attributes, args, kwargs)
            bound = _bind_args(func, args, kwargs)
            with span_factory(span_name, attributes=computed_attrs) as span:
                if apply_pre:
                    apply_pre(span, bound)
                try:
                    result = await func(*args, **kwargs)
                    if apply_post:
                        apply_post(span, result)
                    span.set_status(StatusCode.OK)
                    return result
                except Exception as exc:  # pragma: no cover - passthrough
                    span.record_exception(exc)
                    span.set_status(StatusCode.ERROR, str(exc))
                    raise

        return async_wrapper

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        computed_attrs = _resolve_attributes(attributes, args, kwargs)
        bound = _bind_args(func, args, kwargs)
        with span_factory(span_name, attributes=computed_attrs) as span:
            if apply_pre:
                apply_pre(span, bound)
            try:
                result = func(*args, **kwargs)
                if apply_post:
                    apply_post(span, result)
                span.set_status(StatusCode.OK)
                return result
            except Exception as exc:  # pragma: no cover - passthrough
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, str(exc))
                raise

    return sync_wrapper


def trace_operation(
    name: str | None = None,
    *,
    attributes: AttributeSpec = None,
    capture_io: bool = False,
) -> Callable[[F], F]:
    """Decorator for general-purpose operation spans."""

    def decorator(func: F) -> F:
        span_name = name or f"{func.__module__}.{func.__qualname__}"

        if not capture_io:
            # Simple case: use the utility directly
            return _wrap_with_span(trace_span, span_name, attributes, func)  # type: ignore[return-value]

        # Custom wrapper needed for capture_io to access args/kwargs
        is_async = inspect.iscoroutinefunction(func)

        if is_async:

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                computed_attrs = _resolve_attributes(attributes, args, kwargs)
                with trace_span(span_name, attributes=computed_attrs) as span:
                    try:
                        span.set_attribute("basalt.observe.args_count", len(args))
                        span.set_attribute("basalt.observe.kwargs_count", len(kwargs))
                        result = await func(*args, **kwargs)
                        span.set_attribute("basalt.observe.return_type", type(result).__name__)
                        span.set_status(StatusCode.OK)
                        return result
                    except Exception as exc:  # pragma: no cover - passthrough
                        span.record_exception(exc)
                        span.set_status(StatusCode.ERROR, str(exc))
                        raise

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            computed_attrs = _resolve_attributes(attributes, args, kwargs)
            with trace_span(span_name, attributes=computed_attrs) as span:
                try:
                    span.set_attribute("basalt.observe.args_count", len(args))
                    span.set_attribute("basalt.observe.kwargs_count", len(kwargs))
                    result = func(*args, **kwargs)
                    span.set_attribute("basalt.observe.return_type", type(result).__name__)
                    span.set_status(StatusCode.OK)
                    return result
                except Exception as exc:  # pragma: no cover - passthrough
                    span.record_exception(exc)
                    span.set_status(StatusCode.ERROR, str(exc))
                    raise

        return sync_wrapper  # type: ignore[return-value]

    return decorator


def _extract_first(bound, keys: tuple[str, ...]) -> Any | None:
    if not bound:
        return None
    for key in keys:
        if key in bound.arguments:
            return bound.arguments[key]
    return None


def _serialize_prompt(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except Exception:
        return str(value)


def _extract_completion(result: Any) -> str | None:
    if result is None:
        return None
    if isinstance(result, str):
        return result

    data: dict[str, Any] | None = None
    if isinstance(result, dict):
        data = result
    elif hasattr(result, "model_dump"):
        try:
            data = result.model_dump()
        except Exception:
            data = None
    elif hasattr(result, "dict"):
        try:
            data = result.dict()
        except Exception:
            data = None
    elif hasattr(result, "__dict__"):
        data = vars(result)

    if not data:
        return None

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
            text = choice.get("text")
            if isinstance(text, str):
                return text

    completion = data.get("completion") or data.get("output")
    if isinstance(completion, str):
        return completion
    return None


def _extract_usage(result: Any) -> tuple[int | None, int | None]:
    usage_section: Any | None = None
    if isinstance(result, dict):
        usage_section = result.get("usage")
    elif hasattr(result, "usage"):
        usage_section = result.usage
    elif hasattr(result, "model_dump"):
        try:
            dumped = result.model_dump()
            usage_section = dumped.get("usage")
        except Exception:
            usage_section = None
    if not isinstance(usage_section, dict):
        return None, None
    input_tokens = usage_section.get("prompt_tokens") or usage_section.get("input_tokens")
    output_tokens = usage_section.get("completion_tokens") or usage_section.get("output_tokens")
    input_tokens = int(input_tokens) if isinstance(input_tokens, (int, float)) else None
    output_tokens = int(output_tokens) if isinstance(output_tokens, (int, float)) else None
    return input_tokens, output_tokens


def _apply_llm_request_metadata(span: LLMSpanHandle, bound) -> None:
    if not bound:
        return
    model = _extract_first(bound, ("model", "model_name"))
    if isinstance(model, str):
        span.set_model(model)
    prompt = _extract_first(bound, ("prompt", "input", "inputs", "messages", "question"))
    serialized = _serialize_prompt(prompt)
    if serialized and trace_content_enabled():
        span.set_prompt(serialized)


def _apply_llm_response_metadata(span: LLMSpanHandle, result: Any) -> None:
    completion = _extract_completion(result)
    if completion and trace_content_enabled():
        span.set_completion(completion)
    input_tokens, output_tokens = _extract_usage(result)
    if input_tokens is not None or output_tokens is not None:
        span.set_tokens(input=input_tokens, output=output_tokens)


def trace_llm(
    name: str | None = None,
    *,
    attributes: AttributeSpec = None,
) -> Callable[[F], F]:
    """Decorator specialized for LLM-centric spans."""

    def decorator(func: F) -> F:
        span_name = name or f"{func.__module__}.{func.__qualname__}"

        def apply_pre(span, bound):
            _apply_llm_request_metadata(span, bound)

        def apply_post(span, result):
            _apply_llm_response_metadata(span, result)

        return _wrap_with_span(
            trace_llm_call,
            span_name,
            attributes,
            func,
            apply_pre=apply_pre,
            apply_post=apply_post,
        )  # type: ignore[return-value]

    return decorator


def _extract_http_status(response: Any) -> int | None:
    if response is None:
        return None
    if hasattr(response, "status"):
        status = response.status
        if isinstance(status, int):
            return status
    if hasattr(response, "status_code"):
        status = response.status_code
        if isinstance(status, int):
            return status
    if isinstance(response, dict):
        code = response.get("status") or response.get("status_code")
        if isinstance(code, int):
            return code
    return None


def trace_http(
    name: str | None = None,
    *,
    attributes: AttributeSpec = None,
) -> Callable[[F], F]:
    """Decorator tailored for HTTP client spans."""

    def decorator(func: F) -> F:
        span_name = name or f"{func.__module__}.{func.__qualname__}"
        is_async = inspect.iscoroutinefunction(func)

        # HTTP decorator needs custom timing and status handling
        def _apply_request_attributes(span, bound) -> None:
            method = _extract_first(bound, ("method", "http_method", "verb"))
            url = _extract_first(bound, ("url", "uri", "endpoint"))
            if isinstance(method, str):
                span.set_attribute("http.method", method.upper())
            if isinstance(url, str):
                span.set_attribute("http.url", url)

        def _finalize(span, status: int | None, duration_s: float) -> None:
            span.set_attribute("http.response_time_ms", round(duration_s * 1000, 2))
            if status is not None:
                span.set_attribute("http.status_code", status)
                code = StatusCode.ERROR if status >= 400 else StatusCode.OK
                span.set_status(code)
            else:
                span.set_status(StatusCode.OK)

        if is_async:

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                computed_attrs = _resolve_attributes(attributes, args, kwargs)
                bound = _bind_args(func, args, kwargs)
                start = time.perf_counter()
                with trace_span(span_name, attributes=computed_attrs) as span:
                    _apply_request_attributes(span, bound)
                    try:
                        response = await func(*args, **kwargs)
                        status = _extract_http_status(response)
                        _finalize(span, status, time.perf_counter() - start)
                        return response
                    except Exception as exc:  # pragma: no cover - passthrough
                        span.record_exception(exc)
                        _finalize(span, None, time.perf_counter() - start)
                        raise

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            computed_attrs = _resolve_attributes(attributes, args, kwargs)
            bound = _bind_args(func, args, kwargs)
            start = time.perf_counter()
            with trace_span(span_name, attributes=computed_attrs) as span:
                _apply_request_attributes(span, bound)
                try:
                    response = func(*args, **kwargs)
                    status = _extract_http_status(response)
                    _finalize(span, status, time.perf_counter() - start)
                    return response
                except Exception as exc:  # pragma: no cover - passthrough
                    span.record_exception(exc)
                    _finalize(span, None, time.perf_counter() - start)
                    raise

        return sync_wrapper  # type: ignore[return-value]

    return decorator
