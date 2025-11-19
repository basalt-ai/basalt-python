"""Observability facade for the Basalt SDK."""

from __future__ import annotations

from typing import Any

from opentelemetry import trace as otel_trace

from .api import Observe, observe
from .config import TelemetryConfig
from .context_managers import (
    EvaluatorConfig,
    EventSpanHandle,
    FunctionSpanHandle,
    LLMSpanHandle,
    RetrievalSpanHandle,
    SpanHandle,
    ToolSpanHandle,
    with_evaluators,
)
from .decorators import ObserveKind, evaluate
from .instrumentation import InstrumentationManager
from .processors import BasaltCallEvaluatorProcessor, BasaltContextProcessor
from .trace import Trace
from .trace import trace_api as trace
from .trace_context import (
    TraceExperiment,
    TraceIdentity,
)

__all__ = [
    # High-level API
    "observe",
    "Observe",
    "ObserveKind",
    "evaluate",

    # Low-level API
    "trace",
    "Trace",

    # Config & Types
    "TelemetryConfig",
    "InstrumentationManager",
    "EvaluatorConfig",
    "TraceIdentity",
    "TraceExperiment",

    # Processors
    "BasaltContextProcessor",
    "BasaltCallEvaluatorProcessor",

    # Helpers
    "with_evaluators",

    # Span Handles (Advanced usage)
    "SpanHandle",
    "LLMSpanHandle",
    "RetrievalSpanHandle",
    "ToolSpanHandle",
    "FunctionSpanHandle",
    "EventSpanHandle",
]

_instrumentation = InstrumentationManager()
