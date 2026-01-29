"""
Service A (Caller) - Port 8000
Makes HTTP call to Service B with trace context propagation.
"""

import logging
import os
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from google import genai
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

from basalt import Basalt, TelemetryConfig
from basalt.observability import start_observe

load_dotenv()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

BASALT_API_KEY = os.getenv("BASALT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def build_basalt_client() -> Basalt:
    # Use environment variable for OTLP endpoint or default to localhost
    otlp_endpoint = os.getenv("BASALT_OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317")

    # Create custom exporter with authentication headers
    # Note: insecure=True is used for local/demo purposes
    exporter = OTLPSpanExporter(
        endpoint=otlp_endpoint,
        headers={"authorization": f"Bearer {BASALT_API_KEY}"},
        insecure=True,
        timeout=10,
    )
    telemetry_config = TelemetryConfig(service_name="service-a",
                                       trace_content=True,
                                       enabled_providers=["google_generativeai"],
                                       exporter=[exporter]
                                       )
    basalt_client = Basalt(api_key=BASALT_API_KEY, telemetry_config=telemetry_config)
    return basalt_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.basalt_client = build_basalt_client()

    logger.info("Service A started")
    yield
    logger.info("Service A shutting down")


app = FastAPI(lifespan=lifespan)
HTTPXClientInstrumentor().instrument()
FastAPIInstrumentor.instrument_app(app)


@app.post("/call-service-b")
@start_observe(
    feature_slug="new-test",
    name="service-a-endpoint",
    metadata={"service": "service-a"},
)
async def call_service_b():
    """Call Service B and check if trace context is propagated."""
    current_span = trace.get_current_span()
    span_ctx = current_span.get_span_context()

    logger.info(
        "Service A - Before HTTP call: trace_id=%s, span_id=%s, is_valid=%s",
        format(span_ctx.trace_id, "032x"),
        format(span_ctx.span_id, "016x"),
        span_ctx.is_valid,
    )

    # Simple Gemini call in Service A
    client = genai.Client(api_key=GEMINI_API_KEY)
    gemini_response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents="Say 'Hello from Service A' in exactly 5 words.",
    )
    gemini_text_a = gemini_response.text.strip() if gemini_response.text else None
    logger.info("Service A - Gemini response: %s", gemini_text_a)

    async with httpx.AsyncClient() as client:
        response = await client.post("http://localhost:8001/endpoint")

        sent_traceparent = response.request.headers.get("traceparent")
        logger.info(
            "Service A - After HTTP call: sent traceparent=%s",
            sent_traceparent,
        )

        return {
            "service_a_trace_id": format(span_ctx.trace_id, "032x"),
            "service_a_gemini": gemini_text_a,
            "sent_traceparent": sent_traceparent,
            "service_b_response": response.json(),
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
