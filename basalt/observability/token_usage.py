"""Helpers for propagating GenAI token usage from child spans to generation spans."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from threading import RLock
from typing import Any

from opentelemetry.sdk.trace import Span, SpanProcessor
from opentelemetry.trace import SpanContext

from . import semconv

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

_Key = tuple[int, int]


@dataclass
class _TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None


_lock = RLock()
_active_generation_spans: dict[_Key, Span] = {}
_token_usage: dict[_Key, _TokenUsage] = {}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _make_key(context: SpanContext | None) -> _Key | None:
    if context is None or not context.is_valid:
        return None
    return context.trace_id, context.span_id


def _coerce_tokens(value: Any) -> int | None:
    """Convert assorted token payloads to integers."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            if "." in stripped:
                return int(float(stripped))
            return int(stripped)
        except ValueError:
            return None
    return None


def _record_usage(key: _Key, *, input_tokens: int | None = None, output_tokens: int | None = None) -> None:
    if input_tokens is None and output_tokens is None:
        return
    with _lock:
        usage = _token_usage.setdefault(key, _TokenUsage())
        if input_tokens is not None:
            usage.input_tokens = (usage.input_tokens or 0) + input_tokens
        if output_tokens is not None:
            usage.output_tokens = (usage.output_tokens or 0) + output_tokens


def _known_generation(key: _Key | None) -> bool:
    if key is None:
        return False
    with _lock:
        return key in _active_generation_spans


def _extract_usage_from_mapping(items: Mapping[str, Any], key: _Key) -> None:
    for name, value in items.items():
        _extract_usage_from_attribute(name, value, key)


def _extract_usage_from_attribute(name: Any, value: Any, key: _Key) -> None:
    name_str = str(name)
    candidate = _coerce_tokens(value)
    if candidate is None:
        return

    if name_str == semconv.GenAI.USAGE_INPUT_TOKENS or name_str == semconv._DeprecatedGenAI.USAGE_PROMPT_TOKENS_DEPRECATED:
        _record_usage(key, input_tokens=candidate)
    elif name_str == semconv.GenAI.USAGE_OUTPUT_TOKENS or name_str == semconv._DeprecatedGenAI.USAGE_COMPLETION_TOKENS_DEPRECATED:
        _record_usage(key, output_tokens=candidate)


def _wrap_method(span: Span, method_name: str,
                  wrapper: Callable[[Callable[..., Any], Sequence[Any], MutableMapping[str, Any]], Any]) -> None:
    original = getattr(span, method_name, None)
    if original is None:
        return

    def _wrapped_method(*args: Any, **kwargs: Any) -> Any:
        return wrapper(original, args, kwargs)

    setattr(span, method_name, _wrapped_method)


# ---------------------------------------------------------------------------
# Public API used by context managers and span processor
# ---------------------------------------------------------------------------

def register_generation_span(span: Span) -> _Key | None:
    key = _make_key(span.get_span_context())
    if key is None:
        return None
    with _lock:
        _active_generation_spans[key] = span
        _token_usage.setdefault(key, _TokenUsage())
    return key


def apply_generation_usage(span: Span, key: _Key | None) -> None:
    if key is None:
        return
    with _lock:
        usage = _token_usage.pop(key, None)
        _active_generation_spans.pop(key, None)
    if usage is None:
        return

    attributes = getattr(span, "attributes", None) or {}
    current_input = _coerce_tokens(attributes.get(semconv.GenAI.USAGE_INPUT_TOKENS))
    if current_input is None:
        legacy_input = _coerce_tokens(
            attributes.get(semconv._DeprecatedGenAI.USAGE_PROMPT_TOKENS_DEPRECATED)
        )
        current_input = legacy_input

    current_output = _coerce_tokens(attributes.get(semconv.GenAI.USAGE_OUTPUT_TOKENS))
    if current_output is None:
        legacy_output = _coerce_tokens(
            attributes.get(semconv._DeprecatedGenAI.USAGE_COMPLETION_TOKENS_DEPRECATED)
        )
        current_output = legacy_output

    if usage.input_tokens is not None and current_input is None:
        span.set_attribute(semconv.GenAI.USAGE_INPUT_TOKENS, usage.input_tokens)
    if usage.output_tokens is not None and current_output is None:
        span.set_attribute(semconv.GenAI.USAGE_OUTPUT_TOKENS, usage.output_tokens)


def wrap_child_for_usage(span: Span, parent_key: _Key | None) -> None:
    """Wrap a child span so tokens recorded on it bubble up to its generation parent."""
    if parent_key is None or not _known_generation(parent_key):
        return
    if getattr(span, "_basalt_usage_wrapped", False):
        return

    def attribute_wrapper(original: Callable[..., Any], args: Sequence[Any], kwargs: MutableMapping[str, Any]) -> Any:
        name = None
        value = None
        if args:
            name = args[0]
            if len(args) > 1:
                value = args[1]
        if name is None:
            name = kwargs.get("name") or kwargs.get("key") or kwargs.get("attribute")
        if value is None:
            value = kwargs.get("value")
        if name is not None:
            _extract_usage_from_attribute(name, value, parent_key)
        return original(*args, **kwargs)

    def attributes_wrapper(original: Callable[..., Any], args: Sequence[Any], kwargs: MutableMapping[str, Any]) -> Any:
        if args:
            payload = args[0]
        else:
            payload = kwargs.get("attributes")
        if isinstance(payload, Mapping):
            _extract_usage_from_mapping(payload, parent_key)
        return original(*args, **kwargs)

    _wrap_method(span, "set_attribute", attribute_wrapper)
    _wrap_method(span, "set_attributes", attributes_wrapper)


class TokenUsageSpanProcessor(SpanProcessor):
    """Intercept spans to propagate token usage to parent generation spans."""

    def on_start(self, span: Span, parent_context: Any = None) -> None:
        # Extract SpanContext from Context if possible
        span_context = None
        if parent_context is not None:
            # Try to get the span context from the parent_context if it's a Context
            try:
                from opentelemetry.trace import get_current_span
                parent_span = get_current_span(parent_context)
                if parent_span is not None:
                    span_context = parent_span.get_span_context()
            except Exception:
                span_context = None
        parent_key = _make_key(span_context)
        wrap_child_for_usage(span, parent_key)

    def on_end(self, span) -> None:  # pragma: no cover - nothing to do on span end
        return

    def shutdown(self) -> None:  # pragma: no cover - no resources to release
        with _lock:
            _active_generation_spans.clear()
            _token_usage.clear()

    def force_flush(self, *args: Any, **kwargs: Any) -> bool:  # pragma: no cover - nothing buffered
        return True
