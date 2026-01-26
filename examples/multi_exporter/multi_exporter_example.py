import os

from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as GRPCSpanExporter,
)
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

from basalt import Basalt, TelemetryConfig
from basalt.observability import start_observe

# Configuration
BASALT_API_KEY = os.environ.get("BASALT_API_KEY", "your-basalt-api-key")
BASALT_COLLECTOR_ENDPOINT = "https://grpc.otel.getbasalt.ai"  # Basalt's default collector
LOCAL_COLLECTOR_ENDPOINT = "localhost:4317"  # Local OTel collector


# Create exporters for three destinations

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
telemetry_config = TelemetryConfig(
    service_name="multi-export-demo",
    environment="production",
    exporter=[
        basalt_exporter,  # Export to Basalt for advanced features
        local_exporter,  # Export to local collector
        console_exporter,  # Export to console for debugging
    ],
)

basalt = Basalt(api_key=BASALT_API_KEY, telemetry_config=telemetry_config)


# Simulate a traced workflow


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

# Flush to ensure traces are sent before exit
basalt.shutdown()
