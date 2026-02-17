import json
import logging
import os

from openai import OpenAI
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

from basalt import Basalt
from basalt.observability import Identity, ObserveKind, evaluate, observe, start_observe
from basalt.observability.config import TelemetryConfig

# --- Constants ---
# Use a standard model name that users are likely to have access to
OPENAI_MODEL_NAME = "gpt-4o-mini"


def configure_debug_logging():
    """
    Enable DEBUG-level logging for all relevant OpenTelemetry and Basalt subsystems
    so that exporter, HTTP/gRPC transport, and instrumentation activity is visible.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    )

    debug_loggers = [
        # OTel SDK internals
        "opentelemetry",
        "opentelemetry.sdk",
        "opentelemetry.sdk.trace",
        "opentelemetry.sdk.trace.export",
        # OTLP gRPC exporter + transport
        "opentelemetry.exporter.otlp",
        "opentelemetry.exporter.otlp.proto.grpc",
        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter",
        # gRPC channel / requests
        "grpc",
        # OTel auto-instrumentation providers
        "opentelemetry.instrumentation",
        "opentelemetry.instrumentation.openai",
        # Basalt SDK
        "basalt",
    ]

    for name in debug_loggers:
        logging.getLogger(name).setLevel(logging.DEBUG)


def build_basalt_client():
    """
    Initialize the Basalt client with specific telemetry configurations.
    Exports to both the console (for local debug visibility) and the
    Basalt OTLP collector at otel.getbasalt.dev.
    """
    # 1. Setup API Keys & Endpoints
    basalt_key = os.getenv("BASALT_API_KEY")
    if not basalt_key:
        logging.warning("BASALT_API_KEY not found. Using placeholder.")
        basalt_key = "test-key"

    # 2. Console exporter – prints every span to stdout for live debugging
    console_exporter = ConsoleSpanExporter()

    # 3. Basalt OTLP exporter – sends spans to otel.getbasalt.dev
    basalt_exporter = OTLPSpanExporter(
        endpoint="https://otel.getbasalt.dev",
        headers={"authorization": f"Bearer {basalt_key}"},
        insecure=False,
        timeout=10,
    )

    # 4. Configure Telemetry with BOTH exporters
    telemetry = TelemetryConfig(
        service_name="openai-demo",
        exporter=[console_exporter, basalt_exporter],
        enabled_providers=["openai"],  # Only instrument OpenAI calls
    )

    # 5. Return Client
    return Basalt(
        api_key=basalt_key,
        base_url="https://public-api.getbasalt.dev",
        observability_metadata={
            "env": "development",
            "provider": "openai",
            "example": "auto-instrumentation",
        },
        telemetry_config=telemetry,
    )


# Initialize global client (logging should be configured before this in main, but for module level: careful)
# In a real app, do this inside startup logic.
configure_debug_logging()
basalt_client = build_basalt_client()

# Initialize OpenAI Client securely
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    openai_client = OpenAI(api_key=openai_api_key)
else:
    logging.warning("OPENAI_API_KEY not set. OpenAI calls will be mocked.")
    openai_client = None


@observe(kind=ObserveKind.TOOL, name="get_weather")
def get_weather(location: str):
    """Mock weather tool."""
    # Observe metadata helps track tool usage in traces
    observe.set_metadata({"tool_used": "get_weather", "location": location})
    return json.dumps({"location": location, "temperature": "22C", "condition": "Sunny"})


@evaluate("helpfulness")
def run_weather_assistant(user_query: str, experiment=None):
    # 'start_observe' creates a span for this workflow
    with start_observe(
        name="weather_assistant",
        feature_slug="cocotest",
        experiment=experiment,
        identity=Identity(organization={"id": "123", "name": "Demo Corp"}, user={"id": "456", "name": "Alice"}),
    ) as span:
        span.set_input({"query": user_query})

        # 1. Mock Tool Call
        weather_data = get_weather("San Francisco, CA")

        # 2. LLM Call
        if openai_client is None:
            content = "Mock response: OPENAI_API_KEY is missing."
        else:
            # We wrap the call in a prompt context. Even if we don't explicitly inject the text,
            # Basalt tracks that this specific prompt version was active during the LLM call.
            # We changed the slug to 'weather-system-prompt' to make semantic sense.
            with basalt_client.prompts.get_sync(
                "weather-system-prompt",
                variables={"location": "San Francisco"},
            ):
                response = openai_client.chat.completions.create(
                    model=OPENAI_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a helpful weather assistant."},
                        {
                            "role": "user",
                            "content": f"Context: {weather_data}\n\nQuery: {user_query}",
                        },
                    ],
                )
                content = response.choices[0].message.content

        # 3. Finalize
        # Ensure we always pass a string to the output for consistent logging
        content_str = (content or "")[:100]
        observe.set_output({"response": content_str})

        return content


def main():
    # Debug logging is already configured at module level above; nothing extra needed here.
    logging.debug("main() starting – exporters: ConsoleSpanExporter + OTLPSpanExporter(otel.getbasalt.dev)")

    # Create an experiment to group traces
    experiment = basalt_client.experiments.create_sync(
        feature_slug="cocotest",
        name="Weather Assistant Experiment",
    )
    logging.info(f"Experiment created: id={experiment.id}, name={experiment.name}")

    try:
        # Run the workflow
        result = run_weather_assistant("What's the weather like in SF?", experiment=experiment)
        logging.info(f"Final Result: {result}")
    except Exception as e:
        logging.error(f"Workflow failed: {e}")
    finally:
        # CRITICAL: Shutdown ensures traces are flushed to the exporter before exit
        logging.info("Flushing telemetry...")
        basalt_client.shutdown()


if __name__ == "__main__":
    main()
