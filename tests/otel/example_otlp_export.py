#!/usr/bin/env python
"""
Example script demonstrating how to export OTLP JSON from instrumented LLM calls.

This script shows how to:
1. Set up OTEL instrumentation for an LLM provider
2. Make an LLM API call
3. Export the resulting span(s) as OTLP JSON
4. Print or save the JSON for use in TypeScript tests

Usage:
    # For OpenAI (requires API key)
    export OPENAI_API_KEY="sk-..."
    python tests/otel/example_otlp_export.py openai

    # For Anthropic (requires API key)
    export ANTHROPIC_API_KEY="sk-ant-..."
    python tests/otel/example_otlp_export.py anthropic

    # For Google Gemini (requires API key)
    export GOOGLE_API_KEY="..."
    python tests/otel/example_otlp_export.py google_generativeai
"""

import json
import os
import sys

from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from otlp_utils import spans_to_otel_json

from basalt.observability import InstrumentationManager
from basalt.observability.config import TelemetryConfig


def export_otlp_json_for_provider(provider_name: str, output_file: str | None = None):
    """
    Make an LLM call and export the resulting span as OTLP JSON.

    Args:
        provider_name: Name of the provider (openai, anthropic, google_generativeai, etc.)
        output_file: Optional file path to save JSON. If None, prints to stdout.
    """
    # Set up in-memory span exporter
    exporter = InMemorySpanExporter()

    # Initialize instrumentation
    config = TelemetryConfig(
        enable_llm_instrumentation=True,
        llm_trace_content=True,
        llm_enabled_providers=[provider_name],
    )
    manager = InstrumentationManager()
    manager.initialize(config=config)

    # Attach our exporter to capture spans
    if manager._tracer_provider:
        manager._tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Make LLM call based on provider
    try:
        if provider_name == "openai":
            import openai

            client = openai.OpenAI()
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say hello"}],
                max_tokens=10,
            )

        elif provider_name == "anthropic":
            import anthropic

            client = anthropic.Anthropic()
            client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=10,
                messages=[{"role": "user", "content": "Say hello"}],
            )

        elif provider_name == "google_generativeai":
            import google.generativeai as genai

            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            model = genai.GenerativeModel("gemini-1.5-flash")
            model.generate_content(
                "Say hello",
                generation_config={"max_output_tokens": 10},
            )

        elif provider_name == "vertexai":
            import vertexai
            from vertexai.generative_models import GenerativeModel

            project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

            vertexai.init(project=project_id, location=location)
            model = GenerativeModel("gemini-1.5-flash")
            model.generate_content(
                "Say hello",
                generation_config={"max_output_tokens": 10},
            )

        else:
            sys.exit(1)

    except Exception:
        manager.shutdown()
        sys.exit(1)

    # Get finished spans
    spans = exporter.get_finished_spans()
    if not spans:
        manager.shutdown()
        sys.exit(1)

    # Get the LLM span (usually the last one)
    llm_span = spans[-1]

    # Convert to OTLP JSON
    otlp_json = spans_to_otel_json(
        spans=[llm_span],
        resource_attributes=dict(llm_span.resource.attributes),
        scope_name=llm_span.instrumentation_scope.name,
        scope_version=llm_span.instrumentation_scope.version,
    )

    # Output the JSON
    json_str = json.dumps(otlp_json, indent=2)

    if output_file:
        with open(output_file, "w") as f:
            f.write(json_str)
    else:
        pass

    # Clean up
    manager.shutdown()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    provider = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else None

    export_otlp_json_for_provider(provider, output)
