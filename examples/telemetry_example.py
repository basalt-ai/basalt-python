"""
Example script demonstrating telemetry configuration for the Basalt SDK.

This file is for illustrative purposes only and does not make live API calls.
"""

from __future__ import annotations

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from basalt import Basalt, OpenLLMetryConfig, TelemetryConfig
from basalt.observability.context_managers import trace_llm_call, trace_span
from basalt.observability.decorators import trace_llm, trace_operation


def build_default_client() -> Basalt:
    telemetry = TelemetryConfig(
        service_name="telemetry-example",
        environment="development",
        enable_openllmetry=True,
        openllmetry_config=OpenLLMetryConfig(
            trace_content=False,  # omit prompt/completion bodies
        ),
    )
    return Basalt(api_key="fake-key", telemetry_config=telemetry)


def build_disabled_client() -> Basalt:
    return Basalt(api_key="fake-key", enable_telemetry=False)


def build_custom_exporter_client() -> Basalt:
    exporter = OTLPSpanExporter(endpoint="https://otel.example.com/v1/traces")
    telemetry = TelemetryConfig(service_name="custom-exporter", exporter=exporter)
    return Basalt(api_key="fake-key", telemetry_config=telemetry)


@trace_operation(name="dataset.process", attributes=lambda slug: {"dataset.slug": slug})
def process_dataset(slug: str) -> str:
    # Application logic instrumented automatically.
    return f"processed:{slug}"


@trace_llm(name="llm.generate")
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

    with trace_llm_call("manual.llm") as llm_span:
        llm_span.set_model("gpt-4")
        llm_span.set_prompt("Tell me a joke")
        llm_span.set_completion("Why did the SDK cross the road?")


def main() -> None:
    client = build_default_client()
    disabled_client = build_disabled_client()
    custom_exporter_client = build_custom_exporter_client()

    process_dataset("demo-dataset")
    generate_summary(model="gpt-4", prompt="Explain observability in one line")
    trace_manual_sections()

    client.shutdown()
    disabled_client.shutdown()
    custom_exporter_client.shutdown()


if __name__ == "__main__":
    main()
