"""Tracing package for basalt.

This package provides OpenTelemetry tracing setup and utilities.
"""

from .provider import (
    BasaltConfig,
    create_tracer_provider,
    get_tracer,
    setup_tracing,
    shutdown_tracing,
)

__all__ = [
    "BasaltConfig",
    "create_tracer_provider",
    "get_tracer",
    "setup_tracing",
    "shutdown_tracing",
]
