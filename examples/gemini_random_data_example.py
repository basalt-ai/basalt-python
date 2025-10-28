"""
Example: Gather random data from a public API, then query Gemini (Google AI Studio) with
Basalt telemetry using a custom OTLP exporter.

Requirements:
- `requests` for HTTP calls
- `google-genai` for Gemini (NEW SDK - `from google import genai`)
- `opentelemetry-instrumentation-google-genai` for automatic instrumentation
- Basalt SDK installed

Install with:
    pip install requests google-genai opentelemetry-instrumentation-google-genai basalt-sdk

Note: This example uses the NEW Google GenAI SDK (google-genai).
      For the legacy SDK (google-generativeai), use opentelemetry-instrumentation-google-generativeai instead.
"""

import os

import requests
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from basalt import Basalt, TelemetryConfig
from basalt.observability.decorators import trace_llm


# --- 1. Build Basalt client with custom OTLP exporter ---
def build_custom_exporter_client() -> Basalt:
    exporter = OTLPSpanExporter(endpoint="http://127.0.0.1:4317", insecure=True, timeout=10)
    telemetry = TelemetryConfig(
        service_name="gemini-random-demo",
        exporter=exporter,
        enable_llm_instrumentation=True,  # Enable automatic Gemini instrumentation
    )
    return Basalt(api_key=os.getenv("BASALT_API_KEY", "fake-key"), telemetry_config=telemetry)

basalt_client = build_custom_exporter_client()

# --- 2. Gather random data from a public API ---
def get_random_joke() -> str:
    """Fetch a random joke from the Official Joke API."""
    resp = requests.get("https://official-joke-api.appspot.com/random_joke", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return f"{data['setup']} {data['punchline']}"

# --- 3. Query Gemini (Google AI Studio) with the random data ---
try:
    from google import genai
except ImportError:
    genai = None

@trace_llm(name="gemini.summarize_joke")
def summarize_joke_with_gemini(joke: str) -> str | None:
    """Send the joke to Gemini and get a summary or explanation."""
    if genai is None:
        raise RuntimeError("google-genai is not installed")
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", "fake-key"))
    response = client.models.generate_content(model="gemini-2.5-flash-lite", contents=joke)
    return response.text


def main():
    joke = get_random_joke()
    try:
        summarize_joke_with_gemini(joke)
    except Exception:
        pass
    basalt_client.shutdown()

if __name__ == "__main__":
    main()
