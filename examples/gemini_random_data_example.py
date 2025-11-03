"""
Example: Gather random data from a public API, query Gemini (Google AI Studio), and fetch
a prompt from Basalt's API with full OpenTelemetry instrumentation.

This example demonstrates:
- External HTTP calls with httpx (optionally instrumented)
- Basalt Prompt API integration (internal calls are instrumented)
- Gemini LLM calls with automatic instrumentation
- Custom OTLP exporter configuration

Requirements:
- `httpx` for HTTP calls (optional HTTP client instrumentation)
- `google-genai` for Gemini (NEW SDK - `from google import genai`)
- `opentelemetry-instrumentation-google-genai` for automatic Gemini instrumentation
- `opentelemetry-instrumentation-httpx` (optional) for HTTPX client instrumentation
- Basalt SDK installed

Install with:
    pip install httpx google-genai opentelemetry-instrumentation-google-genai \
                basalt-sdk
    # Optional: add HTTPX instrumentation
    pip install opentelemetry-instrumentation-httpx

Note: This example uses the NEW Google GenAI SDK (google-genai).
      For the legacy SDK (google-generativeai), use opentelemetry-instrumentation-google-generativeai instead.
"""

import logging
import os

import httpx
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from basalt import Basalt, TelemetryConfig
from basalt.observability.context_managers import trace_span
from basalt.observability.decorators import evaluator, trace_retrieval


# --- 1. Build Basalt client with custom OTLP exporter ---
def build_custom_exporter_client() -> Basalt:
    exporter = OTLPSpanExporter(endpoint="http://127.0.0.1:4317", insecure=True, timeout=10)
    telemetry = TelemetryConfig(
        service_name="gemini-demo",
        exporter=exporter,
        enable_llm_instrumentation=True,  # Enable automatic Gemini instrumentation
        llm_trace_content=True,
        llm_enabled_providers=["google_generativeai"],  # Only instrument Gemini calls
    )

    # Initialize Basalt client first (this sets up the TracerProvider)
    client = Basalt(api_key=os.getenv("BASALT_API_KEY", "fake-key"), telemetry_config=telemetry,
                    trace_user={"id": "user-1234"})

    return client

basalt_client = build_custom_exporter_client()

# --- 2. Gather random data from a public API ---
@trace_retrieval(name="http.get_random_joke")
def get_random_joke() -> str:
    """Fetch a random joke from the Official Joke API using httpx (instrumented)."""
    with httpx.Client() as client:
        resp = client.get("https://official-joke-api.appspot.com/random_joke", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        logging.debug(f"Joke API response: {data}")
        return f"{data['setup']} {data['punchline']}"

# --- 3. Fetch a prompt from Basalt API (demonstrates internal instrumentation) ---
def get_prompt_from_basalt(slug: str, joke_text: str, explanation_audience: str) -> str:
    """
    Fetch a prompt from Basalt's Prompt API.

    This call is automatically traced by the Basalt SDK, demonstrating
    internal HTTP call instrumentation.
    """
    try:

        prompt = basalt_client.prompts.get_sync(
            slug,
            variables={
                "jokeText": joke_text,
                "explanationAudience": explanation_audience,
            },
        )
        logging.info(f"Fetched prompt: {prompt.slug} (version: {prompt.version})")
        return prompt.text
    except Exception as exc:
        logging.warning(f"Failed to fetch prompt '{slug}': {exc}")
        raise


# --- 4. Query Gemini (Google AI Studio) with the random data ---
try:
    from google import genai
except ImportError:
    genai = None

@evaluator(
    slugs=["hallucinations", "clarity"],
    sample_rate=1.0,
    metadata=lambda joke, **kwargs: {
        "joke_length": len(joke),
    }
)
def summarize_joke_with_gemini(joke: str) -> str | None:
    """
    Send the joke to Gemini and get a summary or explanation.

    The @evaluator decorator will:
    - Attach evaluator slugs to the auto-instrumented Gemini span
    - Set config with sample_rate to the span
    - Resolve and attach metadata (joke_length) to the span

    All of this happens automatically via the BasaltCallEvaluatorProcessor
    when the Gemini instrumentation creates its span!
    """
    if genai is None:
        raise RuntimeError("google-genai is not installed")
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", "fake-key"))
    response = client.models.generate_content(model="gemini-2.5-flash-lite", contents=joke)
    logging.debug(f"Gemini response: {getattr(response, 'text', response)}")
    return response.text


def main():
    logging.basicConfig(level=logging.DEBUG)

    # Wrap the entire workflow in a single trace span
    with trace_span(
        "workflow.gemini_random_data",
        attributes={
            "workflow.type": "gemini-joke-demo",
            "service": "gemini-random-demo"
        }
    ) as span:

        # 1. Fetch a random joke using httpx (external HTTP call - instrumented)
        joke = get_random_joke()
        span.add_evaluator("joke-quality-check")
        span.set_evaluator_config({"sample_rate": 0.8})

        span.set_input({"joke": joke})

        logging.info(f"Random joke: {joke}")
        span.set_attribute("joke.length", len(joke))

        # 2. Fetch a prompt from Basalt API (internal SDK call - instrumented)
        prompt_slug = os.getenv("BASALT_TEST_PROMPT_SLUG", "joke-analyzer")
        prompt_text = get_prompt_from_basalt(prompt_slug, joke, "a curious geek adult")
        logging.info(f"Basalt prompt preview: {prompt_text[:100]}...")

        # 3. Query Gemini with the joke (LLM call - instrumented)
        try:
            gemini_result = summarize_joke_with_gemini(prompt_text)
            logging.info(f"Gemini summary: {gemini_result}")
            span.set_attribute("gemini.status", "success")
            span.set_attribute("gemini.response_length", len(gemini_result) if gemini_result else 0)
            span.set_output({
                "summary": gemini_result,
                "status": "success",
            })

        except Exception as exc:
            logging.error(f"Gemini error: {exc}")
            span.record_exception(exc)
            span.set_attribute("gemini.status", "error")
            span.set_output({
                "status": "error",
                "error": str(exc),
            })

    # Shutdown and flush telemetry
    logging.info("Shutting down and flushing telemetry...")
    basalt_client.shutdown()

if __name__ == "__main__":
    main()
