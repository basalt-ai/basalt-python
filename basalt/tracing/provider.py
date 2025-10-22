"""OpenTelemetry tracing provider configuration for Basalt SDK."""

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from basalt.config import config as basalt_config


class BasaltConfig:
    """Configuration for Basalt tracing."""

    def __init__(
        self,
        service_name: str = "basalt-sdk",
        service_version: str | None = None,
        environment: str | None = None,
        extra_resource_attributes: dict | None = None,
    ):
       """
        Initialize Basalt tracing configuration.

        Args:
            service_name: The name of the service using the SDK.
            service_version: The version of the service.
            environment: Deployment environment (e.g., 'production', 'staging').
            extra_resource_attributes: Additional OpenTelemetry resource attributes.
        """
       self.service_name = service_name
       self.service_version = service_version or basalt_config.get("sdk_version", "unknown")
       self.environment = environment
       self.extra_resource_attributes = extra_resource_attributes or {}


def create_tracer_provider(
    config: BasaltConfig,
    exporter: SpanExporter | None = None,
) -> TracerProvider:
    """
    Create and configure an OpenTelemetry TracerProvider for Basalt.

    Args:
        config: BasaltConfig instance with service and environment info.
        exporter: Optional SpanExporter. Defaults to ConsoleSpanExporter for debugging.

    Returns:
        A configured TracerProvider instance.
    """
    # Build resource attributes
    resource_attrs = {
        "service.name": config.service_name,
        "service.version": config.service_version,
        "basalt.sdk.type": basalt_config.get("sdk_type", "python"),
        "basalt.sdk.version": basalt_config.get("sdk_version", "unknown"),
    }

    if config.environment:
        resource_attrs["deployment.environment"] = config.environment

    resource_attrs.update(config.extra_resource_attributes)

    resource = Resource.create(resource_attrs)
    provider = TracerProvider(resource=resource)

    exporter = exporter or ConsoleSpanExporter()
    # Use SimpleSpanProcessor for ConsoleSpanExporter and InMemorySpanExporter (for testing)
    # Use BatchSpanProcessor for production exporters
    use_simple = isinstance(exporter, (ConsoleSpanExporter, InMemorySpanExporter))
    processor_cls = SimpleSpanProcessor if use_simple else BatchSpanProcessor
    provider.add_span_processor(processor_cls(exporter))

    return provider


def setup_tracing(
    config: BasaltConfig,
    exporter: SpanExporter | None = None,
) -> TracerProvider:
    """
    Set up global OpenTelemetry tracing for the Basalt SDK.

    Args:
        config: Tracing configuration.
        exporter: Optional SpanExporter to use.

    Returns:
        The configured TracerProvider.
    """
    provider = create_tracer_provider(config, exporter)
    trace.set_tracer_provider(provider)
    return provider

def get_tracer(name: str, version: str | None = None) -> trace.Tracer:
    """
    Get a tracer instance from the global TracerProvider.

    Args:
        name: The name of the tracer (typically the module or component name).
        version: Optional version of the tracer.

    Returns:
        trace.Tracer: A tracer instance.

    Example:
        >>> from basalt.tracing.provider import get_tracer
        >>> tracer = get_tracer(__name__)
        >>> with tracer.start_as_current_span("my-operation"):
        ...     # Your code here
        ...     pass
    """
    return trace.get_tracer(name, version)




def shutdown_tracing():
    """
    Shutdown the global TracerProvider.

    Ensures all spans are exported before application exit.
    """
    provider = trace.get_tracer_provider()

    if isinstance(provider, TracerProvider):
        provider.shutdown()
    elif callable(shutdown := getattr(provider, "shutdown", None)):
        shutdown()
    # If there's no provider or it doesn't support shutdown, silently succeed
    # This can happen in tests or when the provider hasn't been initialized
