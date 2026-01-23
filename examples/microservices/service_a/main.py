"""
Service A - Primary Service

This service demonstrates:
- start_observe with feature_slug "support-ticket"
- HTTP request to Service B
- Distributed tracing across services
- Proper shutdown for telemetry flushing
- Auto-instrumentation for FastAPI and httpx
"""

import logging
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

from basalt import Basalt, TelemetryConfig
from basalt.observability import Observe, ObserveKind, async_start_observe

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global Basalt client
basalt_client: Basalt | None = None

# OpenAI client and configuration
openai_client: AsyncOpenAI | None = None
OPENAI_MODEL = "gpt-4o-mini"

# Service B endpoint
SERVICE_B_URL = os.getenv("SERVICE_B_URL", "http://localhost:8002")


def build_basalt_client() -> Basalt:
    """
    Initialize the Basalt client with local OTLP exporter.

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
    # Note: insecure=True is used for local/demo purposes
    exporter = OTLPSpanExporter(
        endpoint=otlp_endpoint,
        headers={"authorization": f"Bearer {basalt_key}"},
        insecure=True,
        timeout=10,
    )

    # Configure telemetry with OpenAI auto-instrumentation
    telemetry = TelemetryConfig(
        service_name="service-a-orchestrator",
        enabled_providers=["openai"],  # Auto-instrument OpenAI SDK calls
        trace_content=True,  # Capture prompt and completion content
   #     exporter=[exporter],  # Use custom local exporter
    )

    return Basalt(api_key=basalt_key, telemetry_config=telemetry)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown."""
    global basalt_client, openai_client

    # Startup
    logger.info("Starting Service A...")
    basalt_client = build_basalt_client()
    logger.info("Basalt client initialized with OpenAI auto-instrumentation")

    # Initialize OpenAI client
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        openai_client = AsyncOpenAI(api_key=openai_api_key)
        logger.info("OpenAI client initialized")
    else:
        logger.warning("OPENAI_API_KEY not set. LLM generation will be disabled.")

    # Auto-instrument httpx for outgoing HTTP calls
    HTTPXClientInstrumentor().instrument()
    logger.info("HTTPXClientInstrumentor enabled - outgoing HTTP calls will be traced")

    yield

    # Shutdown
    logger.info("Shutting down Service A...")
    
    # Close OpenAI client
    if openai_client:
        await openai_client.close()
        logger.info("OpenAI client closed")
    
    # Uninstrument httpx
    HTTPXClientInstrumentor().uninstrument()
    
    if basalt_client:
        logger.info("Flushing telemetry...")
        basalt_client.shutdown()
        logger.info("Telemetry flushed successfully")


# Create FastAPI app with lifespan management
app = FastAPI(title="Service A - Primary Service", lifespan=lifespan)

# Auto-instrument FastAPI for incoming HTTP requests (distributed tracing!)
# This now works with Basalt thanks to the smart root detection fix
FastAPIInstrumentor.instrument_app(app)
logger.info("FastAPI instrumentation enabled - distributed tracing active")


@Observe(kind=ObserveKind.TOOL, name="call_service_b")
async def call_service_b() -> dict:
    """
    Make HTTP request to Service B for analysis.

    This function handles the HTTP communication and error handling.
    The httpx client is auto-instrumented, so this call will be traced automatically.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            logger.info(f"Calling Service B at {SERVICE_B_URL}/analyze")
            response = await client.get(f"{SERVICE_B_URL}/analyze")
            response.raise_for_status()

            result = response.json()
            logger.info(f"Received response from Service B: {result.get('status', 'unknown')}")
            return result

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from Service B: {e.response.status_code}")
        raise HTTPException(status_code=502, detail=f"Service B returned error: {e.response.status_code}")

    except httpx.RequestError as e:
        logger.error(f"Request error calling Service B: {e}")
        raise HTTPException(status_code=503, detail="Service B is unavailable")

    except Exception as e:
        logger.error(f"Unexpected error calling Service B: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


async def summarize_with_llm(service_b_response: dict) -> dict:
    """
    Use OpenAI to generate a summary of Service B's analysis.
    
    This function demonstrates auto-instrumented OpenAI calls:
    - GENERATION spans created automatically
    - Token usage captured automatically
    - Model name and content tracked
    - Integrated with Basalt prompts API
    """
    if not openai_client:
        logger.warning("OpenAI client not available")
        return {"summary": "LLM unavailable", "error": "API key not set"}

    try:
        # Get prompt from Basalt API
        prompt_cm = await basalt_client.prompts.get(
            slug="joke-analyzer",
            variables={
                "analysis": str(service_b_response.get("analysis", {})),
                "ticket_id": service_b_response.get("ticket_id", "unknown")
            }
        )

        async with prompt_cm as prompt:
            logger.info(f"Retrieved prompt: {prompt.slug} v{prompt.version}")
            
            # Auto-instrumented OpenAI call - GENERATION span created automatically
            response = await openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes support ticket analyses."},
                    {"role": "user", "content": f"Summarize this analysis in 2-3 sentences: {service_b_response}"}
                ],
                temperature=0.7,
                max_tokens=150
            )

            return {
                "summary": response.choices[0].message.content,
                "model": OPENAI_MODEL,
                "tokens_used": response.usage.total_tokens,
                "prompt_slug": prompt.slug,
                "prompt_version": prompt.version
            }

    except Exception as e:
        logger.error(f"Error generating LLM summary: {e}")
        return {"summary": "Error", "error": str(e)}


@app.get("/process-request")
async def process_support_request():
    """
    Main endpoint for processing support requests.

    This demonstrates:
    1. start_observe with feature_slug="support-ticket"
    2. HTTP call to Service B (auto-instrumented via httpx)
    3. Distributed tracing (trace context propagated via HTTP headers)
    4. Proper input/output tracking
    """
    async with async_start_observe(
        name="process_support_request", feature_slug="support-ticket"
    ) as root_span:
        # Set input for observability
        request_data = {"request_type": "support_ticket_processing", "source": "service-a"}
        root_span.set_input(request_data)
        root_span.set_metadata({"service": "service-a", "endpoint": "/process-request"})

        logger.info("Processing support request")

        # Call Service B for analysis
        # Note: httpx automatically propagates OpenTelemetry context via HTTP headers
        # This enables distributed tracing across services
        analysis_result = await call_service_b()

        # Generate LLM summary using OpenAI
        llm_summary = await summarize_with_llm(analysis_result)

        # Prepare response
        response = {
            "status": "completed",
            "request_type": request_data["request_type"],
            "service_b_response": analysis_result,
            "llm_summary": llm_summary,  # OpenAI-generated summary
            "processed_by": "service-a",
        }

        # Set output for observability
        root_span.set_output(response)

        logger.info("Support request processing completed")

        return response


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "service-a"}


@app.get("/")
async def root():
    """Root endpoint with usage instructions."""
    return {
        "service": "Service A - Primary Service",
        "endpoints": {
            "/process-request": "Process a support request (calls Service B)",
            "/health": "Health check",
        },
        "instructions": "Try: curl http://localhost:8001/process-request",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
