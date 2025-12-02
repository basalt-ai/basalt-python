"""Telemetry configuration models for the Basalt SDK."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any

from opentelemetry.sdk.trace.export import SpanExporter

from basalt.config import config as basalt_sdk_config

BoolLike = bool | str | None


def _as_bool(value: BoolLike) -> bool | None:
    """Convert common truthy/falsey string values to bools."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


@dataclass
class TelemetryConfig:
    """
    Centralized configuration for SDK telemetry.

    Instrument Instrumentation
    --------------------------
    When `enable_instrumentation` is True, the SDK automatically instruments provider
    SDKs to capture traces. By default, all available instruments are instrumented.
    You can control which instruments are instrumented using `enabled_providers` and
    `disabled_providers`.

    Supported instruments include LLM providers, vector databases, and AI frameworks:
        LLM Providers:
        - openai: OpenAI API (via opentelemetry-instrumentation-openai)
        - anthropic: Anthropic API (via opentelemetry-instrumentation-anthropic)
        - google_generativeai: Google Generative AI SDK / Gemini (via opentelemetry-instrumentation-google-generativeai)
          Use for: import google.generativeai
        - google_genai: Google GenAI SDK - NEW (instrumentation not yet available on PyPI)
          Will use: from google import genai
          Code ready but package not published yet
        - bedrock: AWS Bedrock (via opentelemetry-instrumentation-bedrock)
        - vertex-ai / vertexai: Google Vertex AI (via opentelemetry-instrumentation-vertexai)
          Both names work as aliases
        - mistralai: Mistral AI (via opentelemetry-instrumentation-mistralai)

        Vector Databases:
        - chromadb: ChromaDB (via opentelemetry-instrumentation-chromadb)
        - pinecone: Pinecone (via opentelemetry-instrumentation-pinecone)
        - qdrant: Qdrant (via opentelemetry-instrumentation-qdrant)

        Frameworks:
        - langchain: LangChain (via opentelemetry-instrumentation-langchain)
        - llamaindex: LlamaIndex (via opentelemetry-instrumentation-llamaindex)

    Notes:
        - The pyproject.toml defines optional dependency extras for the instrumentations above. Some convenience groups
          (e.g. `framework-all`) reference additional packages such as `haystack` even if that extra is not separately
          defined in the optional-dependencies table; consult `pyproject.toml` for the authoritative extras list.
        - The `google_genai` (new GenAI `from google import genai`) instrumentation is mentioned in code but not yet
          published as a stable OpenTelemetry instrumentation on PyPI; `google-generativeai` targets the existing
          `google.generativeai` package.

    Installation:
        Install instrumentation packages using optional dependencies:
            pip install basalt-sdk[openai,anthropic]  # Specific providers
            pip install basalt-sdk[llm-all]            # All LLM providers included in pyproject.toml
            pip install basalt-sdk[all]                # Everything (as defined by the convenience groups)

        See pyproject.toml [project.optional-dependencies] for all available extras.

    Custom Provider Instrumentation
    --------------------------------
    To add instrumentation for providers not listed above, install the appropriate
    OpenTelemetry instrumentation package and instrument it manually:

        from opentelemetry.instrumentation.custom_provider import CustomProviderInstrumentor

        # Initialize Basalt client first
        basalt = Basalt(api_key="your-key", telemetry_config=...)

        # Then instrument your custom provider
        CustomProviderInstrumentor().instrument()

    Connection Error Handling
    -------------------------
    The SDK automatically wraps HTTP OTLP exporters to suppress connection errors
    during span export. This prevents exceptions from propagating when the endpoint
    is unavailable or misconfigured.

    Connection errors are logged at warning level (visible by default) without
    ugly stacktraces. You'll see clean warning messages like:
    "Span export failed (endpoint may be unavailable): ConnectionError: ..."

    The exporter will continue attempting to send spans in subsequent batches,
    and your application will not be interrupted.
    """

    enabled: bool = True
    service_name: str = "basalt-sdk"
    service_version: str | None = basalt_sdk_config.get("sdk_version", "unknown")
    environment: str | None = None
    enable_instrumentation: bool = True
    """Enable automatic instrumentation of provider SDKs."""

    trace_content: bool = True
    """Whether to include prompt and completion content in traces."""

    enabled_providers: list[str] | None = None
    """
    List of specific providers to instrument. If None (default), all available
    providers will be instrumented. Example: ["openai", "anthropic"]
    """

    disabled_providers: list[str] | None = None
    """
    List of providers to explicitly disable. Takes precedence over enabled_providers.
    Example: ["langchain", "llamaindex"]
    """

    exporter: SpanExporter | None = None
    extra_resource_attributes: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> TelemetryConfig:
        """Return a defensive copy of the telemetry configuration."""
        cloned = replace(self)
        cloned.extra_resource_attributes = dict(self.extra_resource_attributes)
        cloned.enabled_providers = list(self.enabled_providers) if self.enabled_providers else None
        cloned.disabled_providers = list(self.disabled_providers) if self.disabled_providers else None
        return cloned

    def with_env_overrides(self) -> TelemetryConfig:
        """
        Return a copy of the configuration with Basalt-specific environment overrides applied.

        Supported environment variables:
            BASALT_TELEMETRY_ENABLED
            BASALT_SERVICE_NAME
            BASALT_ENVIRONMENT
            BASALT_ENABLED_INSTRUMENTS (comma-separated list)
            BASALT_DISABLED_INSTRUMENTS (comma-separated list)
        """
        cfg = self.clone()

        enabled_env = _as_bool(os.getenv("BASALT_TELEMETRY_ENABLED"))
        if enabled_env is not None:
            cfg.enabled = enabled_env

        service_name = os.getenv("BASALT_SERVICE_NAME")
        if service_name:
            cfg.service_name = service_name

        environment = os.getenv("BASALT_ENVIRONMENT")
        if environment:
            cfg.environment = environment

        enabled_instruments = os.getenv("BASALT_ENABLED_INSTRUMENTS")
        if enabled_instruments:
            cfg.enabled_providers = [p.strip() for p in enabled_instruments.split(",") if p.strip()]

        disabled_instruments = os.getenv("BASALT_DISABLED_INSTRUMENTS")
        if disabled_instruments:
            cfg.disabled_providers = [p.strip() for p in disabled_instruments.split(",") if p.strip()]

        if not cfg.service_version:
            # basalt_sdk_config is a mapping defined in `basalt.config` module
            cfg.service_version = basalt_sdk_config.get("sdk_version", "unknown")

        return cfg

    def should_instrument_provider(self, provider: str) -> bool:
        """
        Determine if a specific provider should be instrumented.

        Args:
            provider: The provider name (e.g., "openai", "anthropic", "chromadb")

        Returns:
            True if the provider should be instrumented, False otherwise.
        """
        # Disabled list takes precedence
        if self.disabled_providers and provider in self.disabled_providers:
            return False

        # If enabled_providers is specified, only instrument those
        if self.enabled_providers is not None:
            return provider in self.enabled_providers

        # Otherwise instrument all by default
        return True
