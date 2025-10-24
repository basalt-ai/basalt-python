"""
Deprecated: Monitor resource types have been removed.

Use basalt.observability APIs for OTEL-based observability:
- basalt.observability.init(): configure tracing
- basalt.observability.observe: decorator for function tracing
- basalt.observability.observe_cm: context manager for manual spans
- Observation helper provides methods to add attributes, events, and metadata
"""

raise ImportError(
    "basalt.resources.monitor is deprecated and removed. "
    "Use basalt.observability for OTEL-based tracing instead."
)
