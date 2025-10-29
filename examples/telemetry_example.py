"""
Example script demonstrating telemetry configuration for the Basalt SDK.

This file is for illustrative purposes only and does not make live API calls.

Custom Provider Instrumentation
--------------------------------
To add instrumentation for providers not included by default, install the
appropriate OpenTelemetry instrumentation package and instrument it manually:

    from opentelemetry.instrumentation.custom_provider import CustomProviderInstrumentor

    # Initialize your Basalt client first
    basalt = Basalt(api_key="your-key", telemetry_config=...)

    # Then instrument your custom provider
    CustomProviderInstrumentor().instrument()

Supported providers include: openai, anthropic, google_genai, cohere, bedrock,
vertexai, together, replicate, langchain, llamaindex, haystack
"""

from __future__ import annotations

from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from basalt import Basalt, TelemetryConfig
from basalt.observability.context_managers import trace_generation, trace_span
from basalt.observability.decorators import trace_generation as trace_generation_decorator
from basalt.observability.decorators import trace_span as trace_span_decorator


def build_default_client() -> Basalt:
    """
    Build a client with default telemetry.

    By default, traces are sent to Basalt's OTEL collector:
    - Production: https://otel.getbasalt.ai/v1/traces
    - Development: http://localhost:4318/v1/traces

    This configuration enables LLM provider instrumentation for all available providers.
    """
    telemetry = TelemetryConfig(
        service_name="telemetry-example",
        environment="development",
        enable_llm_instrumentation=True,
        llm_trace_content=False,  # omit prompt/completion bodies from traces
    )
    return Basalt(api_key="fake-key", telemetry_config=telemetry)


def build_disabled_client() -> Basalt:
    """Build a client with all telemetry disabled."""
    return Basalt(api_key="fake-key", enable_telemetry=False)


def build_custom_exporter_client() -> Basalt:
    """Build a client with a custom OTLP exporter endpoint."""
    exporter = OTLPSpanExporter(endpoint="http://127.0.0.1:4317", insecure=True, timeout=10)
    telemetry = TelemetryConfig(service_name="custom-exporter", exporter=exporter)
    return Basalt(api_key="fake-key", telemetry_config=telemetry)


def build_selective_providers_client() -> Basalt:
    """
    Build a client with selective LLM provider instrumentation.

    This example only instruments OpenAI and Anthropic, ignoring other providers.
    """
    telemetry = TelemetryConfig(
        service_name="selective-providers",
        environment="development",
        enable_llm_instrumentation=True,
        llm_trace_content=True,
        llm_enabled_providers=["openai", "anthropic"],  # Only these providers
    )
    return Basalt(api_key="fake-key", telemetry_config=telemetry)


def build_disabled_providers_client() -> Basalt:
    """
    Build a client with specific providers disabled.

    This example instruments all providers except LangChain and LlamaIndex.
    Useful when you want most providers but need to exclude specific frameworks.
    """
    telemetry = TelemetryConfig(
        service_name="disabled-providers",
        environment="development",
        enable_llm_instrumentation=True,
        llm_trace_content=True,
        llm_disabled_providers=["langchain", "llamaindex"],  # Exclude these
    )
    return Basalt(api_key="fake-key", telemetry_config=telemetry)


@trace_span_decorator(
    name="dataset.process",
    variables=lambda bound: {"dataset.slug": bound.arguments.get("slug")},
)
def process_dataset(slug: str) -> str:
    # Application logic instrumented automatically.
    return f"processed:{slug}"

@trace_generation_decorator(name="llm.generate")
def generate_summary(model: str, prompt: str) -> dict:
    # Simulate the response shape of a typical LLM API call.
    return {
        "choices": [{"message": {"content": f"Summary for: {prompt}"}}],
        "usage": {"prompt_tokens": 42, "completion_tokens": 7},
    }


def trace_manual_sections() -> None:
    with trace_span("custom.section", attributes={"feature": "telemetry-demo"}) as span:
        span.add_event("start")
        span.set_attribute("status", "running")
        span.set_input({"section": "manual"})

        with trace_generation("manual.llm") as llm_span:
            llm_span.set_model("gpt-4")
            llm_span.set_prompt("Tell me a joke")
            llm_span.set_completion("Why did the SDK cross the road?")
            llm_span.set_output({"answer": "Why did the SDK cross the road?"})

        span.set_output({"status": span.attributes.get("status")})


def main() -> None:
    """
    Demonstrate various telemetry configuration options.

    Note: In production, you would typically create and use only ONE Basalt client
    throughout your application. This example creates multiple clients to showcase
    different configuration patterns.

    The SDK handles multiple clients gracefully by:
    - Reusing the global OpenTelemetry TracerProvider
    - Checking if instrumentors are already active before re-instrumenting
    - Safely handling shutdown even if shared instrumentation is still in use
    """
    # Example 1: Default client with all providers instrumented
    # client = build_default_client()

    # Example 2: Client with telemetry completely disabled
    # disabled_client = build_disabled_client()

    # Example 3: Client with custom OTLP exporter
    build_custom_exporter_client()

    # Example 4: Client with selective provider instrumentation
    # selective_client = build_selective_providers_client()

    # Example 5: Client with specific providers disabled
    # disabled_providers_client = build_disabled_providers_client()

    # Run some operations to generate traces
    process_dataset("demo-dataset")
    generate_summary(model="gpt-4", prompt="Explain observability in one line")
    trace_manual_sections()

    # Clean up - shutdown all clients to flush traces
    #for c in [client, disabled_client, custom_exporter_client, selective_client, disabled_providers_client]:
    #    c.shutdown()


if __name__ == "__main__":
    main()
