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

    LLM Provider Instrumentation
    -----------------------------
    When `enable_llm_instrumentation` is True, the SDK automatically instruments LLM provider
    SDKs to capture traces. By default, all available LLM providers are instrumented.
    You can control which providers are instrumented using `llm_enabled_providers` and
    `llm_disabled_providers`.

    Supported providers:
        - openai: OpenAI API (via opentelemetry-instrumentation-openai)
        - anthropic: Anthropic API (via opentelemetry-instrumentation-anthropic)
        - google_genai: Google GenAI (via opentelemetry-instrumentation-google-genai)
        - cohere: Cohere API (via opentelemetry-instrumentation-cohere)
        - bedrock: AWS Bedrock (via opentelemetry-instrumentation-bedrock)
        - vertexai: Google Vertex AI (via opentelemetry-instrumentation-vertexai)
        - together: Together AI (via opentelemetry-instrumentation-together)
        - replicate: Replicate (via opentelemetry-instrumentation-replicate)
        - langchain: LangChain (via opentelemetry-instrumentation-langchain)
        - llamaindex: LlamaIndex (via opentelemetry-instrumentation-llamaindex)
        - haystack: Haystack (via opentelemetry-instrumentation-haystack)

    Custom Provider Instrumentation
    --------------------------------
    To add instrumentation for providers not listed above, install the appropriate
    OpenTelemetry instrumentation package and instrument it manually:

        from opentelemetry.instrumentation.custom_provider import CustomProviderInstrumentor

        # Initialize Basalt client first
        basalt = Basalt(api_key="your-key", telemetry_config=...)

        # Then instrument your custom provider
        CustomProviderInstrumentor().instrument()
    """

    enabled: bool = True
    service_name: str = "basalt-sdk"
    service_version: str | None = basalt_sdk_config.get("sdk_version", "unknown")
    environment: str | None = None
    instrument_http: bool = True
    """Retained for backwards compatibility; instrument your HTTP stack manually if needed."""

    enable_llm_instrumentation: bool = False
    """Enable automatic instrumentation of LLM provider SDKs."""

    llm_trace_content: bool = True
    """Whether to include prompt and completion content in LLM traces."""

    llm_enabled_providers: list[str] | None = None
    """
    List of specific LLM providers to instrument. If None (default), all available
    providers will be instrumented. Example: ["openai", "anthropic"]
    """

    llm_disabled_providers: list[str] | None = None
    """
    List of LLM providers to explicitly disable. Takes precedence over llm_enabled_providers.
    Example: ["langchain", "llamaindex"]
    """

    exporter: SpanExporter | None = None
    extra_resource_attributes: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> TelemetryConfig:
        """Return a defensive copy of the telemetry configuration."""
        cloned = replace(self)
        cloned.extra_resource_attributes = dict(self.extra_resource_attributes)
        cloned.llm_enabled_providers = list(self.llm_enabled_providers) if self.llm_enabled_providers else None
        cloned.llm_disabled_providers = list(self.llm_disabled_providers) if self.llm_disabled_providers else None
        return cloned

    def with_env_overrides(self) -> TelemetryConfig:
        """
        Return a copy of the configuration with Basalt-specific environment overrides applied.

        Supported environment variables:
            BASALT_TELEMETRY_ENABLED
            BASALT_SERVICE_NAME
            BASALT_ENVIRONMENT
        """
        config = self.clone()

        enabled_env = _as_bool(os.getenv("BASALT_TELEMETRY_ENABLED"))
        if enabled_env is not None:
            config.enabled = enabled_env

        service_name = os.getenv("BASALT_SERVICE_NAME")
        if service_name:
            config.service_name = service_name

        environment = os.getenv("BASALT_ENVIRONMENT")
        if environment:
            config.environment = environment

        if not config.service_version:
            config.service_version = basalt_sdk_config.get("sdk_version", "unknown")

        return config

    def should_instrument_provider(self, provider: str) -> bool:
        """
        Determine if a specific LLM provider should be instrumented.

        Args:
            provider: The provider name (e.g., "openai", "anthropic")

        Returns:
            True if the provider should be instrumented, False otherwise.
        """
        # Disabled list takes precedence
        if self.llm_disabled_providers and provider in self.llm_disabled_providers:
            return False

        # If enabled_providers is specified, only instrument those
        if self.llm_enabled_providers is not None:
            return provider in self.llm_enabled_providers

        # Otherwise instrument all by default
        return True
