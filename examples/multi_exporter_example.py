"""
Example: Dual Export to Basalt Collector and Datadog Agent.

This example demonstrates how to configure Basalt SDK to export traces to multiple
destinations simultaneously - perfect for customers who want to use Basalt's product
while maintaining their existing observability infrastructure (e.g., Datadog).

Scenario:
---------
A customer has:
- Existing Datadog infrastructure with an agent running with OTel receiver (localhost:4318)
- Wants to use Basalt for advanced LLM observability features
- Needs traces sent to BOTH destinations

Setup:
------
1. Ensure Datadog Agent is running with OTel receiver enabled
2. Have Basalt API key ready
3. Run this script to see dual export in action

Requirements:
-------------
pip install basalt-sdk[openai]  # Includes OpenAI instrumentation
"""

import os

from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as GRPCSpanExporter
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

from basalt import Basalt, TelemetryConfig
from basalt.observability import start_observe

# Configuration
BASALT_API_KEY = os.environ.get("BASALT_API_KEY", "your-basalt-api-key")
BASALT_COLLECTOR_ENDPOINT = "https://grpc.otel.getbasalt.ai"  # Basalt's default collector
LOCAL_COLLECTOR_ENDPOINT = "localhost:4317"  # Local OTel collector

print("=" * 80)
print("Multi-Export Example: Basalt + Local + Console")
print("=" * 80)

# Create exporters for three destinations
print("\n1. Creating exporters...")
print(f"   - Basalt Collector: {BASALT_COLLECTOR_ENDPOINT}")
print(f"   - Local Collector:  {LOCAL_COLLECTOR_ENDPOINT}")
print("   - Console Output:   stdout")

basalt_exporter = GRPCSpanExporter(
    endpoint=BASALT_COLLECTOR_ENDPOINT,
    headers=(("authorization", f"Bearer {BASALT_API_KEY}"),),
)

local_exporter = GRPCSpanExporter(
    endpoint=LOCAL_COLLECTOR_ENDPOINT,
    insecure=True,
    # Local collector typically doesn't need auth headers
)

console_exporter = ConsoleSpanExporter()

# Configure Basalt with THREE exporters
print("\n2. Initializing Basalt SDK with multi-export...")
telemetry_config = TelemetryConfig(
    service_name="multi-export-demo",
    environment="production",
    exporter=[
        basalt_exporter,  # Export to Basalt for advanced features
        local_exporter,   # Export to local collector
        console_exporter, # Export to console for debugging
    ],
    trace_content=True,  # Capture full prompt/response content
    enable_instrumentation=True,  # Auto-instrument OpenAI, etc.
)

basalt = Basalt(api_key=BASALT_API_KEY, telemetry_config=telemetry_config)

print("   ✓ Basalt initialized with multi-export")
print("   ✓ Traces will be sent to Basalt, Local Collector, and Console")

# Simulate a traced workflow
print("\n3. Running traced workflow...")


@start_observe(feature_slug="support-ticket", name="onboard_user")
def onboard_user(user_id: str):
    """Simulate a customer onboarding workflow."""
    from basalt.observability import observe
    
    observe.set_input({"user_id": user_id})
    observe.set_metadata({"workflow_version": "2.1"})

    # In a real app, you might call OpenAI here
    # The auto-instrumentation will capture those calls too
    result = {"status": "success", "user_id": user_id, "onboarding_complete": True}

    observe.set_output(result)
    return result


# Execute the workflow
result = onboard_user("user-12345")
print(f"   ✓ Workflow completed: {result}")

# Flush to ensure traces are sent before exit
print("\n4. Flushing traces to all destinations...")
basalt.shutdown()
print("   ✓ Traces sent!")