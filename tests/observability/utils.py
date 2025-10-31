"""Shared helpers for observability tests."""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from basalt.observability.processors import BasaltCallEvaluatorProcessor, BasaltContextProcessor

_EXPORTER: InMemorySpanExporter | None = None
_PROCESSORS_INSTALLED = False


def get_exporter() -> InMemorySpanExporter:
    """Return a singleton in-memory exporter attached to the global provider."""
    global _EXPORTER, _PROCESSORS_INSTALLED
    if _EXPORTER is None:
        _EXPORTER = InMemorySpanExporter()
        provider = trace.get_tracer_provider()
        if not isinstance(provider, TracerProvider):
            provider = TracerProvider()
            trace.set_tracer_provider(provider)
        provider.add_span_processor(SimpleSpanProcessor(_EXPORTER))
    else:
        provider = trace.get_tracer_provider()
        if not isinstance(provider, TracerProvider):
            return _EXPORTER

    if not _PROCESSORS_INSTALLED:
        provider.add_span_processor(BasaltContextProcessor())
        provider.add_span_processor(BasaltCallEvaluatorProcessor())
        _PROCESSORS_INSTALLED = True
    return _EXPORTER
