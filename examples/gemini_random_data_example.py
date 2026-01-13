import asyncio
import logging
import os

import httpx
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

try:
    from google import genai
except ImportError:
    genai = None

from basalt import Basalt, TelemetryConfig
from basalt.observability import EvaluationConfig, ObserveKind, evaluate, observe, start_observe

# --- Constants ---
# specific model version to ensure consistency across execution and telemetry
GEMINI_MODEL_NAME = "gemini-2.5-flash-lite"


# --- 1. Build Basalt client with custom OTLP exporter ---
def build_custom_exporter_client() -> Basalt:
    """
    Build a Basalt client with a custom OTLP exporter.

    IMPORTANT: When providing a custom exporter, you must manually add authentication
    headers if your collector requires authentication. The SDK only adds headers
    automatically when building the default exporter from environment variables.
    """
    # Get API key for authentication (Fail fast if not set, or use placeholder)
    api_key = os.getenv("BASALT_API_KEY")
    if not api_key:
        logging.warning("BASALT_API_KEY not found. Using placeholder.")
        api_key = "place-holder-key"

    # Use environment variable for OTLP endpoint or default to localhost
    otlp_endpoint = os.getenv("BASALT_OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317")

    # Create a custom exporter with authentication headers
    # Note: insecure=True is used here for local/demo purposes.
    # Use secure credentials for production.
    exporter = OTLPSpanExporter(
        endpoint=otlp_endpoint, headers={"authorization": f"Bearer {api_key}"}, insecure=True, timeout=10
    )

    telemetry = TelemetryConfig(
        service_name="gemini-demo",
        enabled_providers=["google-genai", "google_generativeai"],
        # exporter=[exporter],  # Uncommented to make the example functional
    )

    # Initialize Basalt client first (this sets up the TracerProvider)
    client = Basalt(api_key=api_key, telemetry_config=telemetry)

    return client


basalt_client = build_custom_exporter_client()


# --- 2. Gather random data from a public API ---
@observe(name="http.get_random_joke", kind=ObserveKind.RETRIEVAL)
async def get_random_joke() -> str:
    """Fetch a random joke from the Official Joke API using httpx (instrumented)."""
    async with httpx.AsyncClient() as client:
        resp = await client.get("https://official-joke-api.appspot.com/random_joke", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        logging.debug(f"Joke API response: {data}")
        return f"{data['setup']} {data['punchline']}"


# --- 3. Query Gemini (Google AI Studio) with the random data ---
@evaluate(slugs=["hallucinations", "clarity"])
async def summarize_joke_with_gemini(joke: str) -> str | None:
    """
    Send the joke to Gemini and get a summary or explanation.
    """
    if genai is None:
        raise RuntimeError("google-genai is not installed")

    # Use a dummy key if env var is missing to allow code to 'run' (and fail gracefully)
    # rather than crash on import or setup.
    gemini_key = os.getenv("GEMINI_API_KEY", "dummy-key")

    client = genai.Client(api_key=gemini_key)
    async with client.aio as aclient:
        response = await aclient.models.generate_content(model=GEMINI_MODEL_NAME, contents=joke)

    logging.debug(f"Gemini response: {getattr(response, 'text', response)}")
    return response.text


# --- 4. Main Workflow ---
@start_observe(
    name="workflow.random_data_pipeline",
    feature_slug="support-ticket",
    evaluate_config=EvaluationConfig(sample_rate=0.5),
)
async def start_workflow() -> None:
    # 1. Fetch a random joke using httpx (external HTTP call - instrumented)
    joke = await get_random_joke()

    observe.set_metadata({"joke.length": len(joke)})
    observe.set_input({"joke": joke})

    logging.info(f"Random joke: {joke}")

    # 2. Fetch a prompt from Basalt API (internal SDK call - instrumented)
    prompt_slug = "joke-analyzer"

    # 3. Query Gemini with the joke (LLM call - instrumented)
    try:
        prompt_context_manager = await basalt_client.prompts.get(
            prompt_slug,
            variables={
                "jokeText": joke,
                "explanationAudience": "a curious geek adult",
            },
        )
        async with prompt_context_manager as prompt_context:
            gemini_result = await summarize_joke_with_gemini(prompt_context.text)

            logging.info(f"Gemini summary: {gemini_result}")

            observe.set_metadata(
                {"gemini.status": "success", "gemini.response_length": len(gemini_result) if gemini_result else 0}
            )

            # Use the constant to ensure attributes match the actual model used
            observe.set_attributes({"gemini.model": GEMINI_MODEL_NAME})

            observe.set_output(
                {
                    "summary": gemini_result,
                    "status": "success",
                }
            )

    except Exception as exc:
        logging.error(f"Gemini error: {exc}")
        observe.set_metadata({"gemini.status": "error"})
        observe.set_output(
            {
                "status": "error",
                "error": str(exc),
            }
        )


def main():
    logging.basicConfig(level=logging.DEBUG)

    try:
        asyncio.run(start_workflow())
    except KeyboardInterrupt:
        pass
    finally:
        # Shutdown and flush telemetry
        logging.info("Shutting down and flushing telemetry...")
        basalt_client.shutdown()


if __name__ == "__main__":
    main()
