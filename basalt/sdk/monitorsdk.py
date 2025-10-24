
"""Deprecated module: MonitorSDK has been removed in favor of OTEL-based observability.

Use basalt.observability APIs:
- init(): configure provider, HTTP auto-instrumentation, optional OpenLLMetry
- @observe / observe_cm(): create spans and add custom attributes/events

This module intentionally raises an ImportError to prevent usage.
"""

raise ImportError(
    "basalt.sdk.monitorsdk is deprecated and removed. "
    "Use basalt.observability (init, observe decorator/context manager) instead."
)
