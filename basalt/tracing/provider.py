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
        Initialize Basalt configuration.

        Args:
            service_name: The name of the service using the SDK.
            service_version: The version of the service.
            environment: The environment (e.g., 'production', 'staging', 'development').
            extra_resource_attributes: Additional resource attributes to include.
        """
        self.service_name = service_name
        self.service_version = service_version or basalt_config.get('sdk_version', 'unknown')
        self.environment = environment
        self.extra_resource_attributes = extra_resource_attributes or {}


def create_tracer_provider(
    config: BasaltConfig,
    exporter: SpanExporter | None = None,
) -> TracerProvider:
    """
    Create and configure an OpenTelemetry TracerProvider for Basalt.

    This function sets up the TracerProvider with appropriate resource attributes
    and configures it with the provided or default exporter.

    Args:
        config: BasaltConfig object containing configuration parameters.
        exporter: Optional SpanExporter to use. If None, ConsoleSpanExporter is used for debugging.

    Returns:
        TracerProvider: A configured TracerProvider instance.

    Example:
        >>> from basalt.tracing.provider import BasaltConfig, create_tracer_provider
        >>> config = BasaltConfig(service_name="my-app", environment="production")
        >>> provider = create_tracer_provider(config)
        >>> # Use the provider with OpenTelemetry
        >>> from opentelemetry import trace
        >>> trace.set_tracer_provider(provider)
    """
    # Build resource attributes
    resource_attributes = {
        "service.name": config.service_name,
        "service.version": config.service_version,
        "basalt.sdk.type": basalt_config.get('sdk_type', 'python'),
        "basalt.sdk.version": basalt_config.get('sdk_version', 'unknown'),
    }

    # Add environment if provided
    if config.environment:
        resource_attributes["deployment.environment"] = config.environment

    # Merge in any extra resource attributes
    resource_attributes.update(config.extra_resource_attributes)

    # Create the resource
    resource = Resource.create(resource_attributes)

    # Create the TracerProvider
    provider = TracerProvider(resource=resource)

    # Use ConsoleSpanExporter for debugging if no exporter is provided
    if exporter is None:
        exporter = ConsoleSpanExporter()

    # Add span processor with the exporter
    # Use SimpleSpanProcessor for ConsoleSpanExporter to avoid background export issues
    if isinstance(exporter, ConsoleSpanExporter):
        span_processor = SimpleSpanProcessor(exporter)
    else:
        span_processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(span_processor)

    return provider


def setup_tracing(
    config: BasaltConfig,
    exporter: SpanExporter | None = None,
) -> TracerProvider:
    """
    Set up OpenTelemetry tracing for the Basalt SDK.

    This is a convenience function that creates a TracerProvider and sets it
    as the global tracer provider.

    Args:
        config: BasaltConfig object containing configuration parameters.
        exporter: Optional SpanExporter to use. If None, ConsoleSpanExporter is used for debugging.

    Returns:
        TracerProvider: The configured and globally set TracerProvider instance.

    Example:
        >>> from basalt.tracing.provider import BasaltConfig, setup_tracing
        >>> config = BasaltConfig(service_name="my-app")
        >>> provider = setup_tracing(config)
        >>> # Tracing is now configured and ready to use
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

    This function should be called at the end of the application to ensure
    all pending spans are exported and resources are cleaned up.

    Example:
        >>> from basalt.tracing.provider import shutdown_tracing
        >>> shutdown_tracing()
    """
    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        provider.shutdown()
        return

    shutdown_fn = getattr(provider, "shutdown", None)
    if callable(shutdown_fn):
        shutdown_fn()
    else:
        raise RuntimeError("The global tracer provider does not support shutdown.")
