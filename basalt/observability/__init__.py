"""Observability facade for the Basalt SDK.

Provides an OTEL-first developer experience with:
- init(): tracer provider setup, HTTP auto-instrumentation, optional OpenLLMetry
- observe decorator: easy function-level spans
- observe context manager: manual spans with helpers to add attributes/events
- trace helpers: set user/session/env/tags/metadata on the current trace
- flush(): force-flush span processors for short-lived apps
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from basalt.tracing.provider import BasaltConfig, get_tracer, setup_tracing


def _safe_import(module: str, name: str | None = None):
    try:
        mod = __import__(module, fromlist=[name] if name else [])
        return getattr(mod, name) if name else mod
    except Exception:
        return None


def init(
    app_name: str = "basalt-sdk",
    *,
    environment: str | None = None,
    exporter: Any | None = None,
    enable_openllmetry: bool = False,
    providers: Iterable[str] | None = None,
    instrument_http: bool = True,
) -> None:
    """Initialize OpenTelemetry for Basalt and optional instrumentations.

    Args:
        app_name: Logical service name to appear in traces.
        environment: Optional deployment environment (prod/staging/local).
        exporter: Optional OTEL SpanExporter instance; defaults to console exporter.
        enable_openllmetry: If True, initialize Traceloop/OpenLLMetry and provider instrumentors.
        providers: Iterable of provider names to instrument (e.g., ["openai", "anthropic"]).
        instrument_http: If True, auto-instrument requests and aiohttp-client.
    """
    # 1) Tracer provider setup
    config = BasaltConfig(service_name=app_name, environment=environment)
    setup_tracing(config, exporter=exporter)

    # 2) HTTP auto-instrumentation
    if instrument_http:
        requests_instrumentor_cls = _safe_import(
            "opentelemetry.instrumentation.requests", "RequestsInstrumentor"
        )
        if requests_instrumentor_cls:
            try:
                requests_instrumentor_cls().instrument()
            except Exception:
                pass

        aiohttp_instrumentor_cls = _safe_import(
            "opentelemetry.instrumentation.aiohttp_client", "AioHttpClientInstrumentor"
        )
        if aiohttp_instrumentor_cls:
            try:
                aiohttp_instrumentor_cls().instrument()
            except Exception:
                pass

    # 3) Optional OpenLLMetry initialization
    if enable_openllmetry:
        traceloop_cls = _safe_import("traceloop.sdk", "Traceloop")
        if traceloop_cls:
            try:
                traceloop_cls.init(app_name=app_name)
            except Exception:
                # proceed even if OpenLLMetry init fails
                pass

        # Instrument known providers if requested
        providers = list(providers or ["openai"])  # default to openai if not provided
        for provider in providers:
            try:
                if provider == "openai":
                    openai_instr_cls = _safe_import(
                        "opentelemetry.instrumentation.openai", "OpenAIInstrumentor"
                    )
                    if openai_instr_cls:
                        openai_instr_cls().instrument()
                elif provider == "anthropic":
                    anthropic_instr_cls = _safe_import(
                        "opentelemetry.instrumentation.anthropic", "AnthropicInstrumentor"
                    )
                    if anthropic_instr_cls:
                        anthropic_instr_cls().instrument()
                elif provider == "google_genai":
                    genai_instr_cls = _safe_import(
                        "opentelemetry.instrumentation.google_genai", "GoogleGenAIInstrumentor"
                    )
                    if genai_instr_cls:
                        genai_instr_cls().instrument()
            except Exception:
                # Ignore individual provider failures
                pass


def _apply_attributes(span: Span, attributes: dict[str, Any] | None) -> None:
    if not attributes:
        return
    for k, v in attributes.items():
        span.set_attribute(k, v)


def observe(
    name: str | None = None,
    *,
    attributes: dict[str, Any] | Callable[..., dict[str, Any]] | None = None,
    capture_io: bool = False,
):
    """Decorator to create an observation span for a function.

    Args:
        name: Optional explicit span name; defaults to module.qualname.
        attributes: Optional dict or a callable (args, kwargs) -> dict to attach as span attributes.
        capture_io: If True, attaches simple metadata about args/return (use with care for PII).
    """
    def decorator(func):
        import functools
        import inspect

        span_name = name or f"{func.__module__}.{func.__qualname__}"
        is_async = inspect.iscoroutinefunction(func)

        def compute_attrs(args, kwargs) -> dict[str, Any] | None:
            if callable(attributes):
                try:
                    return attributes(*args, **kwargs)  # type: ignore[misc]
                except Exception:
                    return None
            return attributes  # may be None

        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                tracer = get_tracer(func.__module__)
                with tracer.start_as_current_span(span_name) as span:
                    _apply_attributes(span, compute_attrs(args, kwargs))
                    try:
                        if capture_io:
                            span.set_attribute("basalt.observe.args_count", len(args))
                            span.set_attribute("basalt.observe.kwargs_count", len(kwargs))
                        result = await func(*args, **kwargs)
                        if capture_io:
                            span.set_attribute("basalt.observe.return_type", type(result).__name__)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                tracer = get_tracer(func.__module__)
                with tracer.start_as_current_span(span_name) as span:
                    _apply_attributes(span, compute_attrs(args, kwargs))
                    try:
                        if capture_io:
                            span.set_attribute("basalt.observe.args_count", len(args))
                            span.set_attribute("basalt.observe.kwargs_count", len(kwargs))
                        result = func(*args, **kwargs)
                        if capture_io:
                            span.set_attribute("basalt.observe.return_type", type(result).__name__)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise

            return sync_wrapper

    return decorator


class Observation:
    """Helper to operate on the current observation span."""

    def __init__(self, span: Span):
        self._span = span

    def add_attributes(self, attrs: dict[str, Any]) -> None:
        _apply_attributes(self._span, attrs)

    def event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self._span.add_event(name=name, attributes=attributes)

    # Convenience helpers for common trace-level metadata parity
    def set_user(self, user_id: str) -> None:
        self._span.set_attribute("basalt.user.id", user_id)

    def set_session(self, session_id: str) -> None:
        self._span.set_attribute("basalt.session.id", session_id)

    def set_environment(self, environment: str) -> None:
        self._span.set_attribute("deployment.environment", environment)

    def add_tags(self, tags: Iterable[str]) -> None:
        self._span.set_attribute("basalt.trace.tags", list(tags))

    def add_metadata(self, metadata: dict[str, Any]) -> None:
        for k, v in metadata.items():
            self._span.set_attribute(f"basalt.meta.{k}", v)


@contextmanager
def observe_cm(name: str, attributes: dict[str, Any] | None = None):
    """Context manager to create an observation span.

    Usage:
        with observe_cm("basalt.trace", {"feature_slug": slug}) as obs:
            obs.add_attributes({"tokens.total": 58})
            obs.event("evaluation", {"score": 0.9})
    """
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span(name) as span:
        _apply_attributes(span, attributes)
        yield Observation(span)


def current_span() -> Span | None:
    span = trace.get_current_span()
    return span if span and span.get_span_context().is_valid else None


def set_trace_user(user_id: str) -> None:
    span = current_span()
    if span:
        span.set_attribute("basalt.user.id", user_id)


def set_trace_session(session_id: str) -> None:
    span = current_span()
    if span:
        span.set_attribute("basalt.session.id", session_id)


def set_trace_env(environment: str) -> None:
    span = current_span()
    if span:
        span.set_attribute("deployment.environment", environment)


def add_trace_tags(tags: Iterable[str]) -> None:
    span = current_span()
    if span:
        span.set_attribute("basalt.trace.tags", list(tags))


def add_trace_metadata(metadata: dict[str, Any]) -> None:
    span = current_span()
    if span:
        for k, v in metadata.items():
            span.set_attribute(f"basalt.meta.{k}", v)


def flush() -> None:
    """Force flush span processors without shutting down the provider."""
    provider = trace.get_tracer_provider()
    try:
        provider.force_flush()  # type: ignore[attr-defined]
    except Exception:
        # If provider doesn't support force_flush, ignore
        pass
