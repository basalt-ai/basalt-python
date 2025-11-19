import logging
import os

import httpx
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from basalt import Basalt, TelemetryConfig
from basalt.observability.api import Observe
from basalt.observability.decorators import ObserveKind, evaluate


# --- 1. Build Basalt client with custom OTLP exporter ---
def build_custom_exporter_client() -> Basalt:
    """
    Build a Basalt client with a custom OTLP exporter.

    IMPORTANT: When providing a custom exporter, you must manually add authentication
    headers if your collector requires authentication. The SDK only adds headers
    automatically when building the default exporter from environment variables.
    """
    # Get API key for authentication
    api_key = os.getenv("BASALT_API_KEY", "valid-token")

    # Create a custom exporter with authentication headers
    # For local development without authentication, you can omit the headers parameter
    exporter = OTLPSpanExporter(
        endpoint="http://127.0.0.1:4317",
        headers={"authorization": f"Bearer {api_key}"},  # Add auth headers manually
        insecure=True,
        timeout=10
    )

    telemetry = TelemetryConfig(
        service_name="gemini-demo",
        exporter=exporter,
        enable_llm_instrumentation=True,  # Enable automatic Gemini instrumentation
        llm_trace_content=True,
        llm_enabled_providers=["google_generativeai"],  # Only instrument Gemini calls
    )

    # Initialize Basalt client first (this sets up the TracerProvider)
    client = Basalt(api_key=api_key, telemetry_config=telemetry)

    return client

basalt_client = build_custom_exporter_client()

# --- 2. Gather random data from a public API ---
@Observe(name="http.get_random_joke", kind=ObserveKind.RETRIEVAL)
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

@evaluate(
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

    model_name = "gemini-2.5-flash-lite"
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", "fake-key"))
    response = client.models.generate_content(model=model_name, contents=joke)

    logging.debug(f"Gemini response: {getattr(response, 'text', response)}")
    return response.text


def main():
    logging.basicConfig(level=logging.DEBUG)

    # Wrap the entire workflow in a single trace span with user/org
    # User/org will automatically propagate to all child spans (including Gemini auto-instrumented spans)
    with Observe(
        "workflow.gemini_random_data",
        identity={"organization": "org_123", "user": "user_456"},
        metadata={
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
        prompt_slug = "joke-analyzer"
        prompt_text = get_prompt_from_basalt(prompt_slug, joke, "a curious geek adult")
        #logging.info(f"Basalt prompt preview: {prompt_text[:100]}...")

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
