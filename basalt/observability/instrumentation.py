"""Instrumentation helpers for the Basalt SDK."""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any
from urllib.parse import urlparse

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as OTLPHTTPSpanExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor as OTelSpanProcessor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)

from basalt.config import config as basalt_sdk_config

from . import semconv
from .config import TelemetryConfig
from .processors import (
    BasaltAutoInstrumentationProcessor,
    BasaltCallEvaluatorProcessor,
    BasaltContextProcessor,
)
from .resilient_exporters import ResilientSpanExporter

logger = logging.getLogger(__name__)


def _safe_import(module: str, target: str) -> Any | None:
    """Safely import a target from a module, returning None on failure."""
    try:
        mod = __import__(module, fromlist=[target])
        return getattr(mod, target)
    except Exception as e:
        logger.debug(f"Failed to import {target} from {module}: {e}")
        return None


class BasaltConfig:
    """Configuration for Basalt tracing."""

    def __init__(
        self,
        service_name: str = "basalt-sdk",
        service_version: str | None = None,
        environment: str | None = None,
        extra_resource_attributes: dict | None = None,
    ) -> None:
        """
        Initialize Basalt tracing configuration.

        Args:
            service_name: The name of the service using the SDK.
            service_version: The version of the service.
            environment: Deployment environment (e.g., 'production', 'staging').
            extra_resource_attributes: Additional OpenTelemetry resource attributes.
        """
        self.service_name = service_name
        self.service_version = service_version or basalt_sdk_config.get("sdk_version", "unknown")
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
        semconv.BasaltSDK.TYPE: basalt_sdk_config.get("sdk_type", "python"),
        semconv.BasaltSDK.VERSION: basalt_sdk_config.get("sdk_version", "unknown"),
    }

    if config.environment:
        resource_attrs["deployment.environment"] = config.environment

    resource_attrs.update(config.extra_resource_attributes)

    resource = Resource.create(resource_attrs)
    provider = TracerProvider(resource=resource)

    if exporter is None:
        exporter = ConsoleSpanExporter()
        warnings.warn(
            "No span exporter configured and default Basalt OTEL endpoint unavailable. "
            "Using ConsoleSpanExporter for debugging. For production, configure an exporter "
            "via TelemetryConfig.exporter or set BASALT_OTEL_EXPORTER_OTLP_ENDPOINT environment variable.",
            UserWarning,
            stacklevel=3,
        )

    processor_cls = SimpleSpanProcessor if isinstance(exporter, ConsoleSpanExporter) else BatchSpanProcessor
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

    Note:
        If a TracerProvider is already set globally, this will return the existing
        provider instead of creating a new one to avoid "Overriding of current
        TracerProvider is not allowed" errors.
    """
    # Check if a tracer provider is already set globally
    existing_provider = trace.get_tracer_provider()
    # If it's a real TracerProvider (not the default proxy), reuse it
    if hasattr(existing_provider, 'add_span_processor'):
        logger.debug("Reusing existing global TracerProvider")
        return existing_provider  # type: ignore[return-value]

    # Otherwise create and set a new one
    provider = create_tracer_provider(config, exporter)
    trace.set_tracer_provider(provider)
    return provider


class InstrumentationManager:
    """Central place to coordinate telemetry initialization."""

    def __init__(self) -> None:
        self._initialized = False
        self._api_key: str | None = None
        self._config: TelemetryConfig | None = None
        self._tracer_provider: TracerProvider | None = None
        self._provider_instrumentors: dict[str, Any] = {}
        self._span_processors: list[OTelSpanProcessor] = []

    def initialize(
        self,
        config: TelemetryConfig | None = None,
        *,
        api_key: str | None = None,
    ) -> None:
        """Initialize tracing and instrumentation layers."""
        self._resolve_api_key(api_key)

        if self._initialized:
            return

        effective_config = (config or TelemetryConfig()).with_env_overrides()
        self._config = effective_config

        if not effective_config.enabled:
            self._initialized = True
            return

        exporter = effective_config.exporter or self._build_exporter_from_env()
        basalt_config = BasaltConfig(
            service_name=effective_config.service_name,
            service_version=effective_config.service_version or "",
            environment=effective_config.environment,
            extra_resource_attributes=effective_config.extra_resource_attributes,
        )
        self._tracer_provider = setup_tracing(basalt_config, exporter=exporter)
        if self._tracer_provider:
            self._install_basalt_processors(self._tracer_provider)

        if effective_config.enable_instrumentation:
            self._initialize_instrumentation(effective_config)

        self._initialized = True

    def shutdown(self) -> None:
        """Flush span processors and shutdown instrumentation."""
        if not self._initialized:
            return

        self._uninstrument_providers()

        provider = self._tracer_provider or trace.get_tracer_provider()
        for processor in self._span_processors:
            try:
                processor.shutdown()
            except Exception:
                logger.debug("Error during processor shutdown", exc_info=True)
        self._span_processors = []
        for attr in ("force_flush", "shutdown"):
            method = getattr(provider, attr, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass

        self._initialized = False
        self._tracer_provider = None

    def _build_exporter_from_env(self) -> SpanExporter | None:
        """Build an OTLP exporter from environment variables if configured."""
        # Check for explicit environment variable override first
        endpoint = os.getenv("BASALT_OTEL_EXPORTER_OTLP_ENDPOINT")

        # Fall back to Basalt's default OTEL collector endpoint
        if not endpoint:
            endpoint = basalt_sdk_config.get("otel_endpoint")
            if endpoint:
                logger.info("Using default Basalt OTEL endpoint: %s", endpoint)

        if not endpoint:
            return None

        headers = self._build_exporter_headers()

        try:
            exporter = self._create_otlp_exporter(endpoint, headers=headers)
            logger.info("Basalt: Using OTLP exporter with endpoint: %s", endpoint)
            return exporter
        except Exception as exc:
            warnings.warn(
                f"Failed to create OTLP exporter for endpoint '{endpoint}': {exc}. "
                "Falling back to ConsoleSpanExporter.",
                UserWarning,
                stacklevel=2,
            )
            return None

    def _build_exporter_headers(self) -> dict[str, str] | None:
        api_key = self._resolve_api_key()
        if not api_key:
            return None

        return {"authorization": f"Bearer {api_key}"}

    def _create_otlp_exporter(
        self,
        endpoint: str,
        *,
        headers: dict[str, str] | None,
    ) -> SpanExporter:
        if self._should_use_http_exporter(endpoint):
            exporter = OTLPHTTPSpanExporter(endpoint=endpoint, headers=headers)
            # Wrap HTTP exporter to suppress connection errors during export
            return ResilientSpanExporter(exporter)

        # gRPC exporter handles connection errors internally, no wrapping needed
        return OTLPSpanExporter(endpoint=endpoint, headers=headers)

    @staticmethod
    def _should_use_http_exporter(endpoint: str) -> bool:
        parsed = urlparse(endpoint)
        scheme = parsed.scheme.lower()
        if scheme not in {"http", "https"}:
            return False

        # Check if hostname contains 'grpc' - indicates gRPC endpoint
        if parsed.hostname and 'grpc' in parsed.hostname.lower():
            return False

        if parsed.port == 4317 and parsed.path in {"", "/"}:
            return False

        return True

    def _resolve_api_key(self, candidate: str | None = None) -> str | None:
        if candidate:
            self._api_key = candidate
        elif self._api_key is None:
            env_api_key = os.getenv("BASALT_API_KEY")
            if env_api_key:
                self._api_key = env_api_key
        return self._api_key

    def _instrument_providers(self, config: TelemetryConfig) -> None:
        """
        Instrument specific providers based on configuration.

        This method directly imports and instruments individual provider instrumentors
        instead of using Traceloop.init() which instruments everything globally.

        Args:
            config: Telemetry configuration specifying which providers to instrument.
        """
        # Comprehensive map of supported LLM providers and their instrumentors
        provider_map = {
            # LLM Providers
            "openai": ("opentelemetry.instrumentation.openai", "OpenAIInstrumentor"),
            "anthropic": ("opentelemetry.instrumentation.anthropic", "AnthropicInstrumentor"),
            # NEW Google GenAI SDK (from google import genai)
            "google_genai": ("opentelemetry.instrumentation.google_genai", "GoogleGenAiSdkInstrumentor"),
            # OLD Google Generative AI SDK (import google.generativeai)
            "google_generativeai": ("opentelemetry.instrumentation.google_generativeai",
                                     "GoogleGenerativeAiInstrumentor"),
            "bedrock": ("opentelemetry.instrumentation.bedrock", "BedrockInstrumentor"),
            "vertexai": ("opentelemetry.instrumentation.vertexai", "VertexAIInstrumentor"),
            "vertex-ai": ("opentelemetry.instrumentation.vertexai", "VertexAIInstrumentor"),  # Alias
            "ollama": ("opentelemetry.instrumentation.ollama", "OllamaInstrumentor"),
            "mistralai": ("opentelemetry.instrumentation.mistralai", "MistralAiInstrumentor"),
            # Frameworks
            "langchain": ("opentelemetry.instrumentation.langchain", "LangchainInstrumentor"),
            "llamaindex": ("opentelemetry.instrumentation.llamaindex", "LlamaIndexInstrumentor"),
            # Vector Databases
            "chromadb": ("opentelemetry.instrumentation.chromadb", "ChromaInstrumentor"),
            "pinecone": ("opentelemetry.instrumentation.pinecone", "PineconeInstrumentor"),
            "qdrant": ("opentelemetry.instrumentation.qdrant", "QdrantInstrumentor"),
        }

        for provider_key, (module_name, class_name) in provider_map.items():
            # Skip if already instrumented
            if provider_key in self._provider_instrumentors:
                continue

            # Check if this provider should be instrumented
            if not config.should_instrument_provider(provider_key):
                logger.debug(f"Skipping instrumentation for provider: {provider_key}")
                continue

            # Try to import the instrumentor
            instrumentor_cls = _safe_import(module_name, class_name)
            if not instrumentor_cls:
                logger.debug(
                    f"Provider '{provider_key}' instrumentor not available. "
                    f"Install with: pip install {module_name.replace('.', '-')}"
                )
                continue

            # Instrument the provider
            try:
                instrumentor = instrumentor_cls()
                # Check if already instrumented to avoid double instrumentation
                if hasattr(instrumentor, 'is_instrumented_by_opentelemetry'):
                    if not instrumentor.is_instrumented_by_opentelemetry:
                        instrumentor.instrument()
                        self._provider_instrumentors[provider_key] = instrumentor
                        logger.info(f"Instrumented provider: {provider_key}")
                    else:
                        logger.debug(f"Provider '{provider_key}' already instrumented")
                        self._provider_instrumentors[provider_key] = instrumentor
                else:
                    # Fallback for instrumentors that don't have the property
                    instrumentor.instrument()
                    self._provider_instrumentors[provider_key] = instrumentor
                    logger.info(f"Instrumented provider: {provider_key}")
            except Exception as exc:
                logger.warning(f"Failed to instrument provider '{provider_key}': {exc}")

    def _initialize_instrumentation(self, config: TelemetryConfig) -> None:
        """
        Initialize provider instrumentation.

        Instead of using Traceloop.init() which instruments everything globally,
        this method directly instruments individual providers based on the
        configuration. This gives you fine-grained control over which providers
        are instrumented and reduces unnecessary overhead.

        Args:
            config: Telemetry configuration specifying trace content and provider settings.
        """
        # Set environment variables for third-party OpenTelemetry instrumentors
        # These variables are READ by the instrumentation libraries (openai, anthropic, etc.)
        # and control whether they capture prompts/completions in traces.
        #
        # Why we do this:
        # - Users configure trace_content in TelemetryConfig (clean Python API)
        # - We translate it to environment variables that instrumentors expect
        # - This way users don't need to manually set TRACELOOP_* environment variables
        #
        # Variables set:
        # - TRACELOOP_TRACE_CONTENT: Used by most OpenLLMetry instrumentors
        # - OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: Used by Google GenAI instrumentor
        os.environ["TRACELOOP_TRACE_CONTENT"] = "true" if config.trace_content else "false"
        os.environ[
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
        ] = "true" if config.trace_content else "false"

        # Instrument providers directly without using Traceloop.init()
        self._instrument_providers(config)

    def _install_basalt_processors(self, provider: TracerProvider) -> None:
        if getattr(provider, "_basalt_processors_installed", False):
            return

        processors: list[OTelSpanProcessor] = [
            BasaltContextProcessor(),
            BasaltCallEvaluatorProcessor(),
            BasaltAutoInstrumentationProcessor(),
        ]
        for processor in processors:
            provider.add_span_processor(processor)

        provider._basalt_processors_installed = True  # type: ignore[attr-defined]
        self._span_processors = processors


    def _uninstrument_providers(self) -> None:
        for provider_key, instrumentor in list(self._provider_instrumentors.items()):
            try:
                # Check if it's actually instrumented before trying to uninstrument
                if hasattr(instrumentor, 'is_instrumented_by_opentelemetry'):
                    if instrumentor.is_instrumented_by_opentelemetry:
                        instrumentor.uninstrument()
                        logger.debug(f"Uninstrumented provider: {provider_key}")
                    else:
                        logger.debug(f"Provider '{provider_key}' already uninstrumented")
                else:
                    # Try to uninstrument anyway if we can't check
                    instrumentor.uninstrument()
                    logger.debug(f"Uninstrumented provider: {provider_key}")
            except Exception as exc:
                logger.debug(f"Error uninstrumenting provider '{provider_key}': {exc}")
        self._provider_instrumentors.clear()
