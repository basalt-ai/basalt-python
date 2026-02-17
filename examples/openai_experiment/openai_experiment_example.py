"""
OpenAI Experiment example with Basalt observability.

This example demonstrates:
1. Creating a Basalt Experiment upfront via the Experiments API
2. Running multiple OpenAI calls in a loop, each producing a separate trace
3. Attaching every trace to the same experiment for grouped analysis

Each call to ``start_observe()`` creates a new root span (and therefore a new
trace) because there is no active parent span between loop iterations.  There
is no need to recreate the Basalt client — trace boundaries are determined by
the OpenTelemetry context, not by the client instance.

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
    Initialize the Basalt client with telemetry configuration.

    Only one client should be created for the entire process lifetime.
    The underlying OpenTelemetry TracerProvider is a global singleton —
    calling ``shutdown()`` on a client permanently kills the provider,
    so it must only be called once when the process is exiting.
    """
    basalt_key = os.getenv("BASALT_API_KEY")
    if not basalt_key:
        logging.warning("BASALT_API_KEY not found. Using placeholder.")
        basalt_key = "test-key"

    telemetry = TelemetryConfig(
        service_name="openai-experiment-demo",
        enabled_providers=["openai"],
    )

    return Basalt(
        api_key=basalt_key,
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
) -> str:
    """
    Analyse a single support ticket inside an observed span
    that is attached to the given experiment.

    Each call creates a new root span (and therefore a separate trace)
    because there is no active parent span when this function is invoked.
    """
    with start_observe(
        name="support_ticket_analysis",
        feature_slug="test_feature",
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
    # Step 1: Create a single Basalt client for the entire process
    # ------------------------------------------------------------------
    basalt_client = build_basalt_client()

    # ------------------------------------------------------------------
    # Step 2: Create the experiment
    # ------------------------------------------------------------------
    logging.info("Creating experiment...")

    experiment = basalt_client.experiments.create_sync(
        feature_slug="test_feature",
        name="Support Ticket Analysis Experiment",
    )
    logging.info(f"Experiment created: id={experiment.id}, name={experiment.name}")

    # ------------------------------------------------------------------
    # Step 3: Loop over tickets — each start_observe() creates a new trace
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

        try:
            result = run_single_ticket(
                ticket=ticket,
                experiment=experiment,
                openai_client=openai_client,
            )
            logging.info(f"Result: {result[:80]}...")
        except Exception as e:
            logging.error(f"Ticket analysis failed: {e}")

    logging.info("")
    logging.info("=" * 60)
    logging.info("All tickets processed. Traces attached to experiment: %s", experiment.id)
    logging.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 4: Shutdown ONCE at the very end — flushes all pending traces
    # ------------------------------------------------------------------
    logging.info("Flushing telemetry...")
    basalt_client.shutdown()


if __name__ == "__main__":
    main()
