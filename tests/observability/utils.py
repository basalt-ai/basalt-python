"""Shared helpers for observability tests."""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

_EXPORTER: InMemorySpanExporter | None = None


def get_exporter() -> InMemorySpanExporter:
    """Return a singleton in-memory exporter attached to the global provider."""
    global _EXPORTER
    if _EXPORTER is None:
        _EXPORTER = InMemorySpanExporter()
        provider = trace.get_tracer_provider()
        if not isinstance(provider, TracerProvider):
            provider = TracerProvider()
            trace.set_tracer_provider(provider)
        provider.add_span_processor(SimpleSpanProcessor(_EXPORTER))
    return _EXPORTER
