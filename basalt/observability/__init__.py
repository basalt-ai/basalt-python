"""Observability facade for the Basalt SDK."""

from __future__ import annotations

from .api import (
    AsyncObserve,
    AsyncStartObserve,
    Identity,
    Observe,
    StartObserve,
    async_observe,
    async_start_observe,
    observe,
    start_observe,
)
from .config import TelemetryConfig
from .context_managers import (
    EvaluationConfig,
    EventSpanHandle,
    FunctionSpanHandle,
    LLMSpanHandle,
    RetrievalSpanHandle,
    SpanHandle,
    StartSpanHandle,
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
    "start_observe",
    "StartObserve",
    "async_observe",
    "AsyncObserve",
    "async_start_observe",
    "AsyncStartObserve",
    "ObserveKind",
    "evaluate",
    # Low-level API
    "trace",
    "Trace",
    # Config & Types
    "TelemetryConfig",
    "InstrumentationManager",
    "EvaluationConfig",
    "TraceIdentity",
    "TraceExperiment",
    "Identity",
    # Processors
    "BasaltContextProcessor",
    "BasaltCallEvaluatorProcessor",
    # Helpers
    "with_evaluators",
    # Span Handles (Advanced usage)
    "SpanHandle",
    "StartSpanHandle",
    "LLMSpanHandle",
    "RetrievalSpanHandle",
    "ToolSpanHandle",
    "FunctionSpanHandle",
    "EventSpanHandle",
]

_instrumentation = InstrumentationManager()
