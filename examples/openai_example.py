import json
import os

from openai import OpenAI
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from basalt import Basalt
from basalt.observability import ObserveKind, evaluate, observe, start_observe
from basalt.observability.config import TelemetryConfig

# Ensure API keys are set
if "BASALT_API_KEY" not in os.environ:
    os.environ["BASALT_API_KEY"] = "test-key"
if "OPENAI_API_KEY" not in os.environ:
    pass
    # We don't exit to allow syntax checking, but real run needs key

# Use environment variable for OTLP endpoint or default to localhost
otlp_endpoint = os.getenv("BASALT_OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

exporter = OTLPSpanExporter(
    endpoint=otlp_endpoint,
    headers={"authorization": f"Bearer {os.environ['BASALT_API_KEY']}"},
    insecure=True,
    timeout=10
)

telemetry = TelemetryConfig(
    service_name="gemini-demo",
    exporter=exporter,
    enable_llm_instrumentation=True,  # Enable automatic Gemini instrumentation
    llm_trace_content=True,
    llm_enabled_providers=["google_generativeai", "openai"],  # Only instrument Gemini calls
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

# Initialize OpenAI client
openai_client = OpenAI()


@observe(kind=ObserveKind.TOOL, name="get_weather")
def get_weather(location: str):
    """Mock weather tool."""
    return json.dumps({"location": location, "temperature": "22C", "condition": "Sunny"})

@start_observe(
    name="weather_assistant",
    identity={
        "organization": {"id": "123", "name": "ACME"},
        "user": {"id": "456", "name": "John Doe"}
    },
    metadata={"service": "weather_api"},
)
@evaluate("helpfulness")
def run_weather_assistant(user_query: str):
    observe.input({"query": user_query})

    # 1. Mock Tool Call (simulating a decision to call a tool)
    weather_data = get_weather("San Francisco, CA")

    # 2. Real LLM Call (Auto-instrumented)
    # Basalt automatically captures the span, input (messages), and output (content).
    response = openai_client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "You are a helpful weather assistant."},
            {"role": "user", "content": f"Context: {weather_data}\n\nQuery: {user_query}"}
        ]
    )

    content = response.choices[0].message.content
    observe.output({"response": content[:100]})

    return content

try:
    result = run_weather_assistant("What's the weather like in SF?")
except Exception:
    pass
