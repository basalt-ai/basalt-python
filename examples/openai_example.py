import json
import logging
import os

from openai import AzureOpenAI, OpenAI
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from basalt import Basalt
from basalt.observability import ObserveKind, evaluate, observe, start_observe
from basalt.observability.config import TelemetryConfig

logging.basicConfig(level=logging.INFO)

# Ensure API keys are set
if "BASALT_API_KEY" not in os.environ:
    os.environ["BASALT_API_KEY"] = "test-key"
if "OPENAI_API_KEY" not in os.environ:
    pass
    # We don't exit to allow syntax checking, but real run needs key

# Use environment variable for OTLP endpoint or default to localhost
otlp_endpoint = os.getenv("BASALT_OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

exporter = OTLPSpanExporter(
    endpoint="http://localhost:4317",
    headers={"authorization": f"Bearer {os.environ['BASALT_API_KEY']}"},
    insecure=True,
    timeout=10
)


telemetry = TelemetryConfig(
    service_name="openai-demo",
    exporter=exporter,
    enabled_providers=["openai"],  # Only instrument OpenAI calls
    )


# Initialize Basalt
# Auto-instrumentation for OpenAI is enabled by default when the library is installed.
client = Basalt(
    api_key=os.environ["BASALT_API_KEY"],
    observability_metadata={
        "env": "development",
        "provider": "openai",
        "example": "auto-instrumentation"
    },
    telemetry_config=telemetry
)


openai_api_version = os.environ.get("OPENAI_API_VERSION", "2025-03-01-preview")
azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
openai_api_key = os.environ.get("OPENAI_API_KEY")
# openai_client = AzureOpenAI(api_key=openai_api_key, api_version=openai_api_version, azure_endpoint=azure_endpoint)
openai_client = OpenAI(api_key=openai_api_key)

@observe(kind=ObserveKind.TOOL, name="get_weather")
def get_weather(location: str):
    """Mock weather tool."""
    observe.set_metadata({"tool_used": "get_weather"})
    observe.set_metadata({"location": location})
    return json.dumps({"location": location, "temperature": "22C", "condition": "Sunny"})

@start_observe(
    name="weather_assistant",
    feature_slug="weather_api",
    identity={
        "organization": {"id": "123", "name": "ACME"},
        "user": {"id": "456", "name": "John Doe"}
    },
    metadata={"service_metadata_Start": "weather_api_start"},
)
@evaluate("helpfulness")
def run_weather_assistant(user_query: str):
    observe.set_input({"query": user_query})

    # 1. Mock Tool Call (simulating a decision to call a tool)
    weather_data = get_weather("San Francisco, CA")

    # 2. Real LLM Call (Auto-instrumented)
    # Basalt automatically captures the span, input (messages), and output (content).
    if openai_client is None:
        # If the samples are executed locally without an API key, provide a mock response
        content = "This is a mock response because OPENAI_API_KEY is not set."
    else:
        response = openai_client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are a helpful weather assistant."},
                {"role": "user", "content": f"Context: {weather_data}\n\nQuery: {user_query}"}
            ]
        )

        content = response.choices[0].message.content
    # Ensure we always pass a string slice to the output
    content_str = (content or "")[:100]
    observe.set_output({"response": content_str})

    return content

try:
    result = run_weather_assistant("What's the weather like in SF?")
except Exception:
    pass
