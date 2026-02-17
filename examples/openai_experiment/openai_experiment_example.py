"""
OpenAI Experiment example with Basalt observability.

This example demonstrates:
1. Creating a Basalt Experiment upfront via the Experiments API
2. Running multiple OpenAI calls in a loop, each producing a separate trace
3. Attaching every trace to the same experiment for grouped analysis
4. Re-initializing the Basalt client in each iteration (each client creates a trace)

Uses the "support-ticket" feature from the Gemini example and asks OpenAI
to analyse customer support tickets.

Usage:
    1. Copy .envrc.example to .envrc and fill in your API keys
    2. Run: direnv allow
    3. Run: ./run.sh
"""

import logging
import os

from openai import OpenAI
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from basalt import Basalt
from basalt.observability import Identity, ObserveKind, observe, start_observe
from basalt.observability.config import TelemetryConfig

# --- Constants ---
OPENAI_MODEL_NAME = "gpt-4o-mini"

# Sample support tickets to iterate over — each one produces a distinct trace
SUPPORT_TICKETS = [
    "I can't log into my account. I've tried resetting my password three times but the reset email never arrives.",
    "My order #12345 was supposed to arrive last Monday but the tracking page still says 'in transit'.",
    "The premium plan was billed twice this month. Please refund the duplicate charge immediately.",
    "Your mobile app crashes every time I try to upload a profile picture on Android 14.",
    "I'd like to cancel my subscription but can't find the option anywhere in the settings.",
]


def build_basalt_client() -> Basalt:
    """
    Initialize a fresh Basalt client with telemetry configuration.

    Each call returns a new client instance.  The underlying OpenTelemetry
    TracerProvider is a singleton, so subsequent clients reuse the same
    provider — only the first call installs exporters and processors.
    """
    basalt_key = os.getenv("BASALT_API_KEY")
    if not basalt_key:
        logging.warning("BASALT_API_KEY not found. Using placeholder.")
        basalt_key = "test-key"

    otlp_endpoint = os.getenv("BASALT_OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    exporter = OTLPSpanExporter(
        endpoint=otlp_endpoint,
        headers={"authorization": f"Bearer {basalt_key}"},
        insecure=True,
        timeout=10,
    )

    telemetry = TelemetryConfig(
        service_name="openai-experiment-demo",
        exporter=exporter,
        enabled_providers=["openai"],
    )

    return Basalt(
        api_key=basalt_key,
        observability_metadata={
            "env": "development",
            "provider": "openai",
            "example": "experiment-loop",
        },
        telemetry_config=telemetry,
    )


@observe(kind=ObserveKind.TOOL, name="classify_ticket")
def classify_ticket(ticket: str) -> str:
    """Mock ticket classifier that returns a category."""
    observe.set_metadata({"tool_used": "classify_ticket", "ticket_length": len(ticket)})

    # Simple keyword-based mock classification
    text = ticket.lower()
    if "password" in text or "log in" in text or "login" in text:
        category = "authentication"
    elif "order" in text or "shipping" in text or "transit" in text:
        category = "shipping"
    elif "bill" in text or "charge" in text or "refund" in text:
        category = "billing"
    elif "crash" in text or "bug" in text or "error" in text:
        category = "technical"
    else:
        category = "general"

    return category


def run_single_ticket(
    ticket: str,
    experiment,
    openai_client: OpenAI | None,
    basalt_client: Basalt,
) -> str:
    """
    Analyse a single support ticket inside an observed span
    that is attached to the given experiment.
    """
    with start_observe(
        name="support_ticket_analysis",
        feature_slug="support-ticket",
        experiment=experiment,
        identity=Identity(
            organization={"id": "123", "name": "Demo Corp"},
            user={"id": "456", "name": "Alice"},
        ),
    ) as span:
        span.set_input({"ticket": ticket})

        # 1. Tool call — classify the ticket
        category = classify_ticket(ticket)
        observe.set_metadata({"ticket_category": category})

        # 2. LLM call — generate a draft response (or mock)
        if openai_client is None:
            content = f"Mock response: ticket classified as '{category}'. OPENAI_API_KEY is missing."
        else:
            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful customer-support agent. "
                            "Analyse the support ticket, acknowledge the customer's issue, "
                            "and provide a concise resolution or next steps."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (f"Category: {category}\n\nCustomer ticket:\n{ticket}"),
                    },
                ],
            )
            content = response.choices[0].message.content

        # 3. Finalize
        content_str = (content or "")[:200]
        observe.set_output({"response": content_str, "category": category})

        return content or ""


def main():
    logging.basicConfig(level=logging.INFO)

    logging.info("=" * 60)
    logging.info("OpenAI Experiment Example: support-ticket analysis in a loop")
    logging.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Create the experiment using an initial Basalt client
    # ------------------------------------------------------------------
    logging.info("Creating experiment...")
    init_client = build_basalt_client()

    experiment = init_client.experiments.create_sync(
        feature_slug="support-ticket",
        name="Support Ticket Analysis Experiment",
    )
    logging.info(f"Experiment created: id={experiment.id}, name={experiment.name}")

    # Shut down the initial client — the experiment object is all we need
    init_client.shutdown()

    # ------------------------------------------------------------------
    # Step 2: Loop over tickets, creating a new Basalt client per iteration
    # ------------------------------------------------------------------
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_client: OpenAI | None = None
    if openai_api_key:
        openai_client = OpenAI(api_key=openai_api_key)
    else:
        logging.warning("OPENAI_API_KEY not set. OpenAI calls will be mocked.")

    for i, ticket in enumerate(SUPPORT_TICKETS, start=1):
        logging.info("")
        logging.info(f"--- Ticket {i}/{len(SUPPORT_TICKETS)} ---")
        logging.info(f"Ticket: {ticket[:60]}...")

        # Build a fresh Basalt client for this iteration
        basalt_client = build_basalt_client()

        try:
            result = run_single_ticket(
                ticket=ticket,
                experiment=experiment,
                openai_client=openai_client,
                basalt_client=basalt_client,
            )
            logging.info(f"Result: {result[:80]}...")
        except Exception as e:
            logging.error(f"Ticket analysis failed: {e}")
        finally:
            # Flush traces before moving to the next iteration
            basalt_client.shutdown()

    logging.info("")
    logging.info("=" * 60)
    logging.info("All tickets processed. Traces attached to experiment: %s", experiment.id)
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
