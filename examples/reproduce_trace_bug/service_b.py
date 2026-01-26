"""
Service B (Callee) - Port 8001
Receives HTTP call from Service A.
Demonstrates that @start_observe creates a NEW trace instead of continuing the parent.
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from google import genai
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from basalt import Basalt
from basalt.observability import start_observe

load_dotenv()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

BASALT_API_KEY = os.getenv("BASALT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def build_basalt_client()-> Basalt:
    basalt_client = Basalt(api_key=BASALT_API_KEY)
    return basalt_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.basalt_client = build_basalt_client()
    logger.info("Service B started")
    yield
    logger.info("Service B shutting down")


app = FastAPI(lifespan=lifespan)
FastAPIInstrumentor.instrument_app(app)


@app.post("/endpoint")
@start_observe(
    feature_slug="new-test",
    name="service-b-endpoint",
    metadata={"service": "service-b"},
)
async def endpoint(request: Request):
    """
    Endpoint that receives the call from Service A.

    BUG: Even though traceparent header is received, @start_observe creates
    a NEW trace instead of continuing the parent trace.
    """
    # Log incoming headers
    traceparent = request.headers.get("traceparent")
    tracestate = request.headers.get("tracestate")
    logger.info(
        "Service B - Incoming headers: traceparent=%s, tracestate=%s",
        traceparent,
        tracestate,
    )

    # Log current OTel context (set by @start_observe)
    current_span = trace.get_current_span()
    span_ctx = current_span.get_span_context()
    logger.info(
        "Service B - Current OTel context: trace_id=%s, span_id=%s, is_valid=%s",
        format(span_ctx.trace_id, "032x"),
        format(span_ctx.span_id, "016x"),
        span_ctx.is_valid,
    )

    # Parse the incoming traceparent to compare
    incoming_trace_id = None
    if traceparent:
        parts = traceparent.split("-")
        if len(parts) >= 2:
            incoming_trace_id = parts[1]

    current_trace_id = format(span_ctx.trace_id, "032x")
    traces_match = incoming_trace_id == current_trace_id

    logger.info(
        "Service B - Trace comparison: incoming=%s, current=%s, MATCH=%s",
        incoming_trace_id,
        current_trace_id,
        traces_match,
    )

    # Simple Gemini call in Service B
    client = genai.Client(api_key=GEMINI_API_KEY)
    gemini_response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents="Say 'Hello from Service B' in exactly 5 words.",
    )
    gemini_text_b = gemini_response.text.strip() if gemini_response.text else None
    logger.info("Service B - Gemini response: %s", gemini_text_b)

    return {
        "incoming_traceparent": traceparent,
        "service_b_trace_id": current_trace_id,
        "service_b_gemini": gemini_text_b,
        "traces_match": traces_match,
        "bug_reproduced": not traces_match,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
