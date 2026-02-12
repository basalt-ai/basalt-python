"""
OpenAI v2 example using responses.parse() with Basalt observability.

This example demonstrates:
1. Using OpenAI's new responses.parse() API for structured outputs
2. Auto-instrumentation of OpenAI calls with Basalt
3. Multiple exporters: Basalt OTLP + Console for debugging

Usage:
    1. Copy .envrc.example to .envrc and fill in your API keys
    2. Run: direnv allow
    3. Run: ./run.sh
"""

import logging
import os
from typing import Optional

from openai import OpenAI
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from pydantic import BaseModel, Field

from basalt import Basalt
from basalt.observability import Identity, start_observe
from basalt.observability.config import TelemetryConfig

# --- Pydantic Models for Structured Output ---


class CalendarEvent(BaseModel):
    """Structured output for calendar event extraction."""

    name: str = Field(description="Name or title of the event")
    date: str = Field(description="Date of the event")
    participants: list[str] = Field(description="List of people participating in the event")


class ProductDetails(BaseModel):
    """Structured output for product information extraction."""

    name: str = Field(description="Name of the product")
    price: float | None = Field(None, description="Price of the product if mentioned")
    category: str = Field(description="Category of the product")
    features: list[str] = Field(default_factory=list, description="Key features or highlights")


# --- Constants ---
OPENAI_MODEL_NAME = "gpt-4o-mini"


def build_basalt_client():
    """
    Initialize the Basalt client with Basalt OTLP + Console exporters.
    """
    # 1. Setup API Keys
    basalt_key = os.getenv("BASALT_API_KEY")
    if not basalt_key:
        logging.warning("BASALT_API_KEY not found. Using placeholder.")
        basalt_key = "test-key"

    # 2. Configure Exporters
    # Primary: Basalt OTLP exporter for the cloud platform
    basalt_exporter = OTLPSpanExporter(
        endpoint="https://grpc.otel.getbasalt.ai",
        headers={"authorization": f"Bearer {basalt_key}"},
        insecure=False,
        timeout=10,
    )

    # Secondary: Console exporter for debugging spans locally
    console_exporter = ConsoleSpanExporter()

    # 3. Configure Telemetry with OpenAI auto-instrumentation
    telemetry = TelemetryConfig(
        service_name="openai-v2-demo",
        exporter=[basalt_exporter, console_exporter],  # Multiple exporters
        enabled_providers=["openai"],  # Only instrument OpenAI calls
    )

    # 4. Return Client
    return Basalt(
        api_key=basalt_key,
        observability_metadata={
            "env": "development",
            "provider": "openai",
            "example": "responses-parse-v2",
            "note": "Using new responses.parse() API",
        },
        telemetry_config=telemetry,
    )


# Initialize global client
basalt_client = build_basalt_client()

# Initialize OpenAI Client securely
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    openai_client = OpenAI(api_key=openai_api_key)
else:
    logging.warning("OPENAI_API_KEY not set. OpenAI calls will be mocked.")
    openai_client = None


@start_observe(feature_slug="calendar-extractor", name="extract_calendar_event")
def extract_calendar_event(user_input: str) -> Optional[CalendarEvent]:
    """
    Extract calendar event information from natural language text.
    Uses OpenAI's new responses.parse() API for structured outputs.
    """
    if openai_client is None:
        logging.warning("OpenAI client not initialized. Returning mock data.")
        return CalendarEvent(name="Mock Event", date="Friday", participants=["Alice", "Bob"])

    # Use the new responses.parse() API
    # This automatically validates the output against the Pydantic model
    response = openai_client.responses.parse(
        model=OPENAI_MODEL_NAME,
        input=[
            {"role": "system", "content": "Extract the event information from the user's message. Return the event name, date, and participants."},
            {"role": "user", "content": user_input},
        ],
        text_format=CalendarEvent,
    )

    # The parsed response is already validated as a Pydantic model
    event = response.output_parsed
    logging.info(f"Extracted event: {event}")

    return event


@start_observe(feature_slug="product-extractor", name="extract_product_details")
def extract_product_details(product_description: str) -> Optional[ProductDetails]:
    """
    Extract product information from a description.
    Uses OpenAI's new responses.parse() API for structured outputs.
    """
    if openai_client is None:
        logging.warning("OpenAI client not initialized. Returning mock data.")
        return ProductDetails(name="Mock Product", category="Electronics", features=["Mock feature"])

    response = openai_client.responses.parse(
        model=OPENAI_MODEL_NAME,
        input=[
            {"role": "system", "content": "Extract product information from the description. Identify the name, price, category, and key features."},
            {"role": "user", "content": product_description},
        ],
        text_format=ProductDetails,
    )

    product = response.output_parsed
    logging.info(f"Extracted product: {product}")

    return product


def main():
    logging.basicConfig(level=logging.INFO)

    logging.info("=" * 60)
    logging.info("OpenAI v2 Example: responses.parse() with Basalt")
    logging.info("=" * 60)
    logging.info("")

    try:
        # Example 1: Calendar Event Extraction
        logging.info("Example 1: Calendar Event Extraction")
        logging.info("-" * 40)

        calendar_input = "Alice and Bob are going to a science fair on Friday at the convention center."
        logging.info(f"Input: {calendar_input}")

        event = extract_calendar_event(calendar_input)
        if event:
            logging.info(f"Event Name: {event.name}")
            logging.info(f"Date: {event.date}")
            logging.info(f"Participants: {', '.join(event.participants)}")

        logging.info("")

        # Example 2: Product Details Extraction
        logging.info("Example 2: Product Details Extraction")
        logging.info("-" * 40)

        product_input = (
            "The new MacBook Pro 16-inch with M3 Max chip is priced at $3499. "
            "It features a stunning Liquid Retina XDR display, 22-hour battery life, "
            "and supports up to 128GB of unified memory. Perfect for professional video editing."
        )
        logging.info(f"Input: {product_input}")

        product = extract_product_details(product_input)
        if product:
            logging.info(f"Product Name: {product.name}")
            logging.info(f"Category: {product.category}")
            logging.info(f"Price: ${product.price}" if product.price else "Price: Not mentioned")
            logging.info(f"Features: {', '.join(product.features)}")

        logging.info("")
        logging.info("=" * 60)
        logging.info("Examples completed successfully!")
        logging.info("Check the console output above for span details.")
        logging.info("=" * 60)

    except Exception as e:
        logging.error(f"Workflow failed: {e}")
        raise
    finally:
        # CRITICAL: Shutdown ensures traces are flushed to the exporters before exit
        logging.info("")
        logging.info("Flushing telemetry...")
        basalt_client.shutdown()


if __name__ == "__main__":
    main()
