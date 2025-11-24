"""Mistral + Basalt Observability Example (Updated Client).

Demonstrates manual instrumentation of a Mistral chat completion using the new
`mistralai` client API (post-migration from deprecated `MistralClient`). The
flow wraps a workflow span and an LLM generation span with explicit input/output
metadata for Basalt observability.

Environment Variables:
    BASALT_API_KEY   - Basalt API key (falls back to test-key).
    MISTRAL_API_KEY  - Mistral API key (required for real responses).
    MISTRAL_MODEL    - Optional model alias/name (e.g. 'small', 'tiny').
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence

from mistralai import Mistral, UserMessage
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from basalt import Basalt
from basalt.observability import ObserveKind, observe, start_observe
from basalt.observability.config import TelemetryConfig

# ---------------------------------------------------------------------------
# Environment bootstrap
# ---------------------------------------------------------------------------
if "BASALT_API_KEY" not in os.environ:
    os.environ["BASALT_API_KEY"] = "test-key"

API_KEY = os.environ.get("MISTRAL_API_KEY", "")  # Empty -> likely to error for real call

# Use environment variable for OTLP endpoint or default to localhost
OTLP_ENDPOINT = os.environ.get("BASALT_OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

exporter = OTLPSpanExporter(
    endpoint=OTLP_ENDPOINT,
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

# ---------------------------------------------------------------------------
# Basalt client initialization
# ---------------------------------------------------------------------------
client = Basalt(
    api_key=os.environ["BASALT_API_KEY"],
    observability_metadata={
        "env": "staging",
        "provider": "mistral",
        "example": "manual-instrumentation"
    },
    telemetry_config=telemetry
)

# ---------------------------------------------------------------------------
# Mistral client (new API)
# ---------------------------------------------------------------------------
mistral_client = Mistral(api_key=API_KEY)


def resolve_model(raw: str) -> str:
    """Resolve a user-provided model alias to a full model name."""
    aliases = {
        "small": "mistral-small-latest",
        "tiny": "mistral-tiny",
        "open-7b": "open-mistral-7b",
        "7b": "open-mistral-7b",
        "large": "mistral-large-latest",
    }
    return aliases.get(raw, raw) if raw else "mistral-small-latest"


def run_mistral_flow(topic: str) -> str:
    """Execute a simple observed workflow that queries Mistral.

    Parameters:
        topic: Subject to explain.

    Returns:
        Assistant response string (or fallback message if unavailable).
    """
    with start_observe(
        name="mistral_workflow",
        identity={
            "organization": {"id": "123", "name": "ACME"},
            "user": {"id": "456", "name": "John Doe"}
        },
        metadata={"provider": "mistral", "model_type": "chat"},
    ):
        observe.input({"topic": topic})

        with observe(name="mistral_chat", kind=ObserveKind.GENERATION) as llm_span:
            model = resolve_model(os.environ.get("MISTRAL_MODEL", ""))
            # Prefer typed message helper for compatibility with the SDK
            messages: Sequence[UserMessage] = [
                UserMessage(content=f"Explain {topic} in one concise sentence.")
            ]

            llm_span.set_input({"messages": messages})
            llm_span.set_attribute("llm.model", model)
            llm_span.set_attribute("llm.provider", "mistral")

            try:
                response = mistral_client.chat.complete(model=model, messages=list(messages))
                # Some responses may return complex content objects; convert to string safely
                raw_content = response.choices[0].message.content
                content = str(raw_content) if raw_content is not None else "<no content>"
                llm_span.set_output(content)
                if getattr(response, "usage", None):
                    llm_span.set_attribute("llm.usage.total_tokens", response.usage.total_tokens)
                return content
            except Exception as exc:  # noqa: BLE001
                fallback = f"Mistral call failed: {exc}"[:250]
                llm_span.set_output(fallback)
                llm_span.set_attribute("error", str(exc))
                return fallback


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    result = run_mistral_flow("Quantum Physics")
    logging.info("Result: %s", result)


if __name__ == "__main__":
    main()
