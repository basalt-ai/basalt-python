"""
Example of using Basalt with Google's Gemini AI for a complete RAG workflow:
1. Fetch data from external API
2. Generate summary with Gemini LLM
3. Create embeddings from the summary

Demonstrates both decorator-based and manual context manager instrumentation.
"""

import asyncio
import logging
import os
from typing import TypedDict

import httpx
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

try:
    from google import genai
except ImportError:
    genai = None

from basalt import Basalt, TelemetryConfig
from basalt.observability import (
    AsyncObserve,
    EvaluationConfig,
    ObserveKind,
    evaluate,
    observe,
    start_observe,
)

# --- Constants ---
# specific model version to ensure consistency across execution and telemetry
GEMINI_MODEL_NAME = "gemini-2.5-flash-lite"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"


class EmbeddingResult(TypedDict):
    """Result of embedding generation."""

    dimension: int
    sample_values: list[float]
    status: str


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
        endpoint=otlp_endpoint,
        headers={"authorization": f"Bearer {api_key}"},
        insecure=True,
        timeout=10,
    )

    telemetry = TelemetryConfig(
        service_name="gemini-demo",
        enabled_providers=["google_generativeai"],
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


async def embed_joke_summary(summary: str) -> EmbeddingResult:
    """
    Generate embeddings for the joke summary using Gemini Embedding model.

    This demonstrates MANUAL instrumentation with AsyncObserve context manager,
    in contrast to the decorator-based approach used in other functions.

    It also showcases the GenAI convenience methods (set_model, set_tokens, etc.)
    that are now available on the base SpanHandle class, eliminating the need
    for type casting or isinstance checks.
    """
    if genai is None:
        raise RuntimeError("google-genai is not installed. Run: pip install google-genai")

    gemini_key = os.getenv("GEMINI_API_KEY", "dummy-key")

    # Manual instrumentation using AsyncObserve context manager
    async with AsyncObserve(
        name="genai.embed_content",
        kind=ObserveKind.GENERATION,
    ) as span:
        try:
            # Create embeddings using the Google GenAI SDK
            client = genai.Client(api_key=gemini_key)
            async with client.aio as aclient:
                result = await aclient.models.embed_content(
                    model=GEMINI_EMBEDDING_MODEL,
                    contents=summary,
                )

            # Extract embedding data
            if result.embeddings is None or len(result.embeddings) == 0:
                raise ValueError("No embeddings returned from API response")
            embedding_vector = result.embeddings[0].values
            if embedding_vector is None:
                raise ValueError("Embedding vector is None from API response")
            dimension_count = len(embedding_vector)

            # Token usage: prefer API metadata, fall back to a simple estimate
            usage_metadata = getattr(result, "usage_metadata", None)
            if isinstance(usage_metadata, dict):
                usage_lookup = usage_metadata
            else:
                usage_lookup = usage_metadata or {}
            input_tokens = (
                usage_lookup.get("prompt_token_count")
                if isinstance(usage_lookup, dict)
                else getattr(usage_lookup, "prompt_token_count", None)
            )
            output_tokens = (
                usage_lookup.get("candidates_token_count")
                if isinstance(usage_lookup, dict)
                else getattr(usage_lookup, "candidates_token_count", None)
            )
            if output_tokens is None:
                output_tokens = (
                    usage_lookup.get("total_token_count")
                    if isinstance(usage_lookup, dict)
                    else getattr(usage_lookup, "total_token_count", None)
                )
            if input_tokens is None:
                input_tokens = len(summary.split()) if summary else 0
            if output_tokens is None:
                output_tokens = 0

            response_model = getattr(result, "model", GEMINI_EMBEDDING_MODEL)

            # Set input/output for telemetry
            span.set_input({"text": summary})

            # Set GenAI semantic convention attributes using convenience methods
            span.set_operation_name("embeddings")  # gen_ai.operation.name
            span.set_provider("google")  # gen_ai.provider.name
            span.set_model(GEMINI_EMBEDDING_MODEL)  # gen_ai.request.model
            span.set_response_model(response_model)  # gen_ai.response.model
            span.set_tokens(input=input_tokens, output=output_tokens)  # gen_ai.usage.*_tokens

            # Set embedding-specific attribute (custom semantic convention)
            span.set_attribute("gen_ai.embeddings.dimension.count", dimension_count)

            # Set metadata for Basalt tracking
            span.set_metadata(
                {
                    "embedding.dimension": dimension_count,
                    "embedding.status": "success",
                    "embedding.model": GEMINI_EMBEDDING_MODEL,
                }
            )

            # Set output (partial vector only - don't log full 768-dim vector)
            output_data: EmbeddingResult = {
                "dimension": dimension_count,
                "sample_values": embedding_vector[:5]
                if embedding_vector
                else [],  # First 5 values only
                "status": "success",
            }
            span.set_output(str(output_data))

            logging.info(
                "Generated %s-dimensional embedding vector",
                dimension_count,
            )
            return output_data

        except Exception as exc:
            logging.error(f"Embedding error: {exc}")
            span.set_metadata(
                {
                    "embedding.status": "error",
                    "embedding.error": str(exc),
                }
            )
            span.set_output(
                {
                    "status": "error",
                    "error": str(exc),
                }
            )
            raise


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
                {
                    "gemini.status": "success",
                    "gemini.response_length": len(gemini_result) if gemini_result else 0,
                }
            )

            # Use the constant to ensure attributes match the actual model used
            observe.set_attributes({"gemini.model": GEMINI_MODEL_NAME})

            observe.set_output(
                {
                    "summary": gemini_result,
                    "status": "success",
                }
            )

            # 4. Generate embeddings for the summary (demonstrates manual instrumentation)
            if gemini_result:
                try:
                    embedding_data = await embed_joke_summary(gemini_result)
                    logging.info(
                        f"Embedding: {embedding_data['dimension']} dimensions, "
                        f"sample: {embedding_data['sample_values'][:3]}"
                    )

                    observe.set_metadata(
                        {
                            "embedding.dimension": embedding_data["dimension"],
                            "embedding.generated": True,
                        }
                    )
                except Exception as exc:
                    logging.error(f"Failed to generate embeddings: {exc}")
                    observe.set_metadata(
                        {
                            "embedding.generated": False,
                            "embedding.error": str(exc),
                        }
                    )
                    # Don't re-raise - allow workflow to continue even if embeddings fail

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
