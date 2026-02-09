"""
Service B - Analysis Service

This service demonstrates:
- start_observe with feature_slug "support-ticket"
- Nested observe span with kind=RETRIEVAL
- Prompt retrieval using prompts API (joke-analyzer)
- Proper shutdown for telemetry flushing
- Auto-instrumentation for FastAPI
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from google import genai
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from basalt import Basalt, TelemetryConfig
from basalt.observability import ObserveKind, async_observe, async_start_observe

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global Basalt client
basalt_client: Basalt | None = None

# Gemini model configuration
GEMINI_MODEL = "gemini-2.5-flash-lite"


def build_basalt_client() -> Basalt:
    """
    Initialize the Basalt client with local OTLP exporter and Gemini instrumentation.

    Uses custom OTLP exporter to avoid conflicts with FastAPI instrumentation.
    Based on pattern from examples/gemini_random_data_example.py.
    """
    # Get API key for authentication
    basalt_key = os.getenv("BASALT_API_KEY")
    if not basalt_key:
        logger.warning("BASALT_API_KEY not found. Using placeholder.")
        basalt_key = "test-key"

    # Use environment variable for OTLP endpoint or default to localhost
    otlp_endpoint = os.getenv("BASALT_OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317")

    # Create custom exporter with authentication headers
    exporter = OTLPSpanExporter(
        endpoint=otlp_endpoint,
        headers={"authorization": f"Bearer {basalt_key}"},
        insecure=True,
        timeout=10,
    )

    # Configure telemetry with Gemini auto-instrumentation
    telemetry = TelemetryConfig(
        service_name="service-b-analysis",
        enabled_providers=["google_generativeai"],  # NEW Google GenAI SDK (from google import genai)
        trace_content=True,  # Capture prompt and completion content
        #   exporter=[exporter],  # Use custom local exporter
    )

    return Basalt(api_key=basalt_key, telemetry_config=telemetry)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown."""
    global basalt_client

    # Startup
    logger.info("Starting Service B...")
    basalt_client = build_basalt_client()
    logger.info("Basalt client initialized for production")

    yield

    # Shutdown
    logger.info("Shutting down Service B...")
    if basalt_client:
        logger.info("Flushing telemetry...")
        basalt_client.shutdown()
        logger.info("Telemetry flushed successfully")


# Create FastAPI app with lifespan management
app = FastAPI(title="Service B - Analysis Service", lifespan=lifespan)

# Auto-instrument FastAPI for incoming HTTP requests (distributed tracing!)
# This now works with Basalt thanks to the smart root detection fix
FastAPIInstrumentor.instrument_app(app)
logger.info("FastAPI instrumentation enabled - distributed tracing active")


async def perform_retrieval(query: str) -> dict:
    """
    Simulate a vector database retrieval operation.

    This function demonstrates the RETRIEVAL observe kind with proper
    span configuration including query, results_count, and top_k.
    """
    async with async_observe(kind=ObserveKind.RETRIEVAL, name="retrieve_ticket_context_in_b") as span:
        # Set retrieval query
        span.set_input(query)

        # Simulate retrieval from vector database
        # In a real application, this would query a vector DB like Pinecone, Qdrant, etc.
        simulated_results = [
            {"id": "doc-1", "content": "Previous support ticket about billing", "score": 0.95},
            {"id": "doc-2", "content": "FAQ about account management", "score": 0.87},
            {"id": "doc-3", "content": "Knowledge base article on troubleshooting", "score": 0.82},
        ]

        # Set retrieval metadata
        span.set_top_k(5)
        span.set_metadata({"retrieval_type": "vector_search", "index": "support-tickets"})

        logger.info(f"Retrieved {len(simulated_results)} results for query: {query}")

        return {"query": query, "results": simulated_results, "count": len(simulated_results)}


async def analyze_with_prompt(ticket_data: dict, context: dict) -> dict:
    """
    Analyze ticket using Gemini LLM with the joke-analyzer prompt.

    This function demonstrates auto-instrumented Gemini calls:
    - GENERATION spans created automatically
    - Token usage captured automatically
    - Model name and content tracked
    - Integrated with Basalt prompts API
    """
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        logger.warning("GEMINI_API_KEY not set. Using fallback.")
        return {"analysis": "Fallback analysis (API key not set)", "error": "GEMINI_API_KEY not configured"}

    try:
        # Retrieve prompt from Basalt
        prompt_cm = await basalt_client.prompts.get(
            slug="joke-analyzer", variables={"ticket_id": ticket_data.get("ticket_id", "unknown"), "context_count": context["count"]}
        )

        async with prompt_cm as prompt:
            logger.info(f"Retrieved prompt: {prompt.slug} v{prompt.version}")

            # Auto-instrumented Gemini call - GENERATION span created automatically
            client = genai.Client(api_key=gemini_api_key)
            async with client.aio as aclient:
                response = await aclient.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=f"""Analyze this support ticket:

Ticket ID: {ticket_data.get("ticket_id", "unknown")}
Context retrieved: {context["count"]} documents

Provide:
1. Sentiment analysis
2. Priority level (low/medium/high)
3. Brief analysis (2-3 sentences)

Context: {context.get("results", [])}
""",
                )

            # Extract response and metadata
            analysis_text = response.text

            # Get token usage if available
            usage_metadata = getattr(response, "usage_metadata", None)
            tokens_used = None
            if usage_metadata:
                tokens_used = {
                    "prompt_tokens": getattr(usage_metadata, "prompt_token_count", 0),
                    "completion_tokens": getattr(usage_metadata, "candidates_token_count", 0),
                    "total_tokens": getattr(usage_metadata, "total_token_count", 0),
                }

            return {
                "prompt_slug": prompt.slug,
                "prompt_version": prompt.version,
                "analysis": analysis_text,
                "model": GEMINI_MODEL,
                "tokens_used": tokens_used,
                "context_used": context["count"],
            }

    except Exception as e:
        logger.error(f"Error with Gemini analysis: {e}")
        return {
            "analysis": "Fallback analysis (Gemini failed)",
            "error": str(e),
        }


@app.get("/analyze")
async def analyze_ticket():
    """
    Main endpoint for ticket analysis.

    This demonstrates:
    1. Get prompt BEFORE start_observe (tests feature_slug propagation fix)
    2. start_observe with feature_slug="support-ticket"
    3. Nested observe span with kind=RETRIEVAL
    4. Prompt retrieval using the prompts API
    5. Proper input/output tracking
    """
    # Get prompt BEFORE start_observe - this tests that feature_slug propagates
    # correctly to the prompt request span even without an active trace context
    pre_prompt_cm = await basalt_client.prompts.get(slug="joke-analyzer", variables={"request_type": "ticket_analysis"})
    async with pre_prompt_cm as pre_prompt:
        logger.info(f"Pre-trace prompt retrieved: {pre_prompt.slug}")

    async with async_start_observe(name="analyze_support_ticket", feature_slug="support-ticket") as root_span:
        # Set input for observability
        ticket_data = {"ticket_id": "DEMO-001", "request_type": "analysis"}
        root_span.set_input(ticket_data)
        root_span.set_metadata({"service": "service-b", "endpoint": "/analyze"})

        logger.info(f"Analyzing ticket: {ticket_data['ticket_id']}")

        # Step 1: Perform retrieval
        retrieval_context = await perform_retrieval("support ticket analysis context")

        # Step 2: Analyze using prompt
        analysis = await analyze_with_prompt(ticket_data, retrieval_context)

        # Prepare response
        response = {
            "status": "success",
            "ticket_id": ticket_data["ticket_id"],
            "analysis": analysis,
            "retrieval_context_count": retrieval_context["count"],
        }

        # Set output for observability
        root_span.set_output(response)

        logger.info(f"Analysis completed for ticket: {ticket_data['ticket_id']}")

        return response


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "service-b"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
