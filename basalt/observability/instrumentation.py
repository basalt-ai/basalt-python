"""Instrumentation helpers for the Basalt SDK."""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)

from basalt.config import config as basalt_sdk_config

from .config import OpenLLMetryConfig, TelemetryConfig

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
        "basalt.sdk.type": basalt_sdk_config.get("sdk_type", "python"),
        "basalt.sdk.version": basalt_sdk_config.get("sdk_version", "unknown"),
    }

    if config.environment:
        resource_attrs["deployment.environment"] = config.environment

    resource_attrs.update(config.extra_resource_attributes)

    resource = Resource.create(resource_attrs)
    provider = TracerProvider(resource=resource)

    if exporter is None:
        exporter = ConsoleSpanExporter()
        warnings.warn(
            "No span exporter configured. Using ConsoleSpanExporter for debugging. "
            "For production, configure an exporter via TelemetryConfig.exporter or set "
            "BASALT_OTEL_EXPORTER_OTLP_ENDPOINT environment variable.",
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
    """
    provider = create_tracer_provider(config, exporter)
    trace.set_tracer_provider(provider)
    return provider


class InstrumentationManager:
    """Central place to coordinate telemetry initialization."""

    def __init__(self) -> None:
        self._initialized = False
        self._config: TelemetryConfig | None = None
        self._tracer_provider: TracerProvider | None = None
        self._http_instrumented = False
        self._requests_instrumentor: Any | None = None
        self._aiohttp_instrumentor: Any | None = None
        self._provider_instrumentors: dict[str, Any] = {}

    def initialize(self, config: TelemetryConfig | None = None) -> None:
        """Initialize tracing and instrumentation layers."""
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

        if effective_config.instrument_http:
            self._instrument_http()

        openll_config = effective_config.resolved_openllmetry()
        if openll_config:
            self._initialize_openllmetry(openll_config)

        self._initialized = True

    def shutdown(self) -> None:
        """Flush span processors and shutdown instrumentation."""
        if not self._initialized:
            return

        self._uninstrument_http()
        self._uninstrument_llm()

        provider = self._tracer_provider or trace.get_tracer_provider()
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
        endpoint = os.getenv("BASALT_OTEL_EXPORTER_OTLP_ENDPOINT")
        if not endpoint:
            return None
        try:
            exporter = OTLPSpanExporter(endpoint=endpoint)
            # Info message rather than warning since this is expected configuration
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

    def _instrument_http(self) -> None:
        if self._http_instrumented:
            return

        requests_cls = _safe_import("opentelemetry.instrumentation.requests", "RequestsInstrumentor")
        if requests_cls:
            try:
                self._requests_instrumentor = requests_cls()
                self._requests_instrumentor.instrument()
            except Exception:
                self._requests_instrumentor = None

        aiohttp_cls = _safe_import(
            "opentelemetry.instrumentation.aiohttp_client", "AioHttpClientInstrumentor"
        )
        if aiohttp_cls:
            try:
                self._aiohttp_instrumentor = aiohttp_cls()
                self._aiohttp_instrumentor.instrument()
            except Exception:
                self._aiohttp_instrumentor = None

        self._http_instrumented = bool(self._requests_instrumentor or self._aiohttp_instrumentor)

    def _instrument_llm_providers(self) -> None:
        provider_map = {
            "openai": ("opentelemetry.instrumentation.openai", "OpenAIInstrumentor"),
            "anthropic": ("opentelemetry.instrumentation.anthropic", "AnthropicInstrumentor"),
            "google_genai": ("opentelemetry.instrumentation.google_genai", "GoogleGenAIInstrumentor"),
        }
        for key, module_info in provider_map.items():
            if key in self._provider_instrumentors:
                continue
            cls = _safe_import(*module_info)
            if not cls:
                continue
            try:
                instrumentor = cls()
                instrumentor.instrument()
                self._provider_instrumentors[key] = instrumentor
            except Exception:
                continue

    def _initialize_openllmetry(self, config: OpenLLMetryConfig) -> None:
        os.environ["TRACELOOP_TRACE_CONTENT"] = "true" if config.trace_content else "false"
        os.environ["TRACELOOP_TELEMETRY"] = "true" if config.telemetry_enabled else "false"

        traceloop_cls = _safe_import("traceloop.sdk", "Traceloop")
        if traceloop_cls:
            init_kwargs: dict[str, Any] = {"app_name": config.app_name}
            if config.disable_batch:
                init_kwargs["disable_batch"] = True
            if config.api_endpoint:
                init_kwargs["otlp_endpoint"] = config.api_endpoint
            if config.headers:
                init_kwargs["headers"] = config.headers
            try:
                traceloop_cls.init(**init_kwargs)
            except Exception:
                pass

        self._instrument_llm_providers()

    def _uninstrument_http(self) -> None:
        for instrumentor in (self._requests_instrumentor, self._aiohttp_instrumentor):
            if instrumentor:
                try:
                    instrumentor.uninstrument()
                except Exception:
                    pass
        self._requests_instrumentor = None
        self._aiohttp_instrumentor = None
        self._http_instrumented = False

    def _uninstrument_llm(self) -> None:
        for instrumentor in self._provider_instrumentors.values():
            try:
                instrumentor.uninstrument()
            except Exception:
                pass
        self._provider_instrumentors.clear()
