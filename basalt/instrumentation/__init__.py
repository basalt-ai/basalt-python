"""
Instrumentation package for the Basalt SDK.

This package provides automatic instrumentation for various AI/ML libraries,
capturing telemetry data and traces for monitoring and observability.
"""
from .openai import OpenAIInstrumentor

__all__ = ["OpenAIInstrumentor"]
