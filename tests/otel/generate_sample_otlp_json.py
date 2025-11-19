#!/usr/bin/env python
"""
Generate sample OTLP JSON from mock LLM spans for testing TypeScript parsers.

Usage:
    python tests/otel/generate_sample_otlp_json.py [provider_name]

    provider_name: openai, anthropic, google_generativeai, or vertexai
                   (default: openai)
"""

import sys

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind
from otlp_utils import spans_to_otel_json


def get_span_config(provider_name: str):
    """Get provider-specific span configuration."""
    configs = {
        "openai": {
            "scope_name": "opentelemetry.instrumentation.openai",
            "scope_version": "0.21.5",
            "span_name": "chat",
            "attributes": {
                "gen_ai.system": "openai",
                "gen_ai.request.model": "gpt-4o-mini",
                "gen_ai.response.model": "gpt-4o-mini",
                "gen_ai.usage.input_tokens": 10,
                "gen_ai.usage.output_tokens": 5,
                "llm.usage.total_tokens": 15,
            },
        },
        "anthropic": {
            "scope_name": "opentelemetry.instrumentation.anthropic",
            "scope_version": "0.21.5",
            "span_name": "chat",
            "attributes": {
                "gen_ai.system": "anthropic",
                "gen_ai.request.model": "claude-3-5-haiku-20241022",
                "gen_ai.response.model": "claude-3-5-haiku-20241022",
                "gen_ai.usage.input_tokens": 10,
                "gen_ai.usage.output_tokens": 5,
            },
        },
        "google_generativeai": {
            "scope_name": "opentelemetry.instrumentation.google_generativeai",
            "scope_version": "0.21.5",
            "span_name": "gemini.generate_content",
            "attributes": {
                "gen_ai.system": "google",
                "gen_ai.request.model": "gemini-1.5-flash",
                "gen_ai.response.model": "gemini-1.5-flash",
                "gen_ai.usage.prompt_tokens": 10,
                "gen_ai.usage.completion_tokens": 5,
                "llm.usage.total_tokens": 15,
            },
        },
        "vertexai": {
            "scope_name": "opentelemetry.instrumentation.vertexai",
            "scope_version": "0.21.5",
            "span_name": "vertexai.generate_content",
            "attributes": {
                "gen_ai.system": "vertex_ai",
                "gen_ai.request.model": "gemini-1.5-flash",
                "gen_ai.response.model": "gemini-1.5-flash",
                "gen_ai.usage.prompt_tokens": 10,
                "gen_ai.usage.completion_tokens": 5,
                "llm.usage.total_tokens": 15,
            },
        },
    }
    return configs[provider_name]


def main():
    provider_name = sys.argv[1] if len(sys.argv) > 1 else "openai"

    if provider_name not in ["openai", "anthropic", "google_generativeai", "vertexai"]:
        sys.exit(1)

    # Get provider configuration
    config = get_span_config(provider_name)

    # Create TracerProvider with realistic resource attributes
    resource = Resource.create(
        {
            "service.name": "llm-service",
            "service.version": "1.0.0-rc1",
            "telemetry.sdk.language": "python",
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.version": "1.38.0",
            "basalt.sdk.type": "python",
            "basalt.sdk.version": "1.0.0-rc1",
        }
    )

    provider = TracerProvider(resource=resource)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Create span with provider-specific attributes
    tracer = provider.get_tracer(config["scope_name"], config["scope_version"])
    with tracer.start_as_current_span(config["span_name"], kind=SpanKind.CLIENT) as span:
        for key, value in config["attributes"].items():
            span.set_attribute(key, value)

    # Get finished span
    py_span = exporter.get_finished_spans()[0]

    # Convert to OTLP JSON
    spans_to_otel_json(
        spans=[py_span],
        resource_attributes=dict(py_span.resource.attributes),
        scope_name=py_span.instrumentation_scope.name,
        scope_version=py_span.instrumentation_scope.version,
    )

    # Print formatted JSON


if __name__ == "__main__":
    main()
