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
class OpenLLMetryConfig:
    """Configuration for Traceloop/OpenLLMetry integration."""

    app_name: str | None = None
    disable_batch: bool = False
    trace_content: bool = True
    api_endpoint: str | None = None
    headers: dict[str, str] | None = None
    telemetry_enabled: bool = True

    def clone(self) -> OpenLLMetryConfig:
        """Return a defensive copy of the configuration."""
        copied = replace(self)
        copied.headers = dict(self.headers) if self.headers else None
        return copied


@dataclass
class TelemetryConfig:
    """Centralized configuration for SDK telemetry."""

    enabled: bool = True
    service_name: str = "basalt-sdk"
    service_version: str | None = basalt_sdk_config.get("sdk_version", "unknown")
    environment: str | None = None
    instrument_http: bool = True
    enable_openllmetry: bool = False
    openllmetry_config: OpenLLMetryConfig | None = None
    exporter: SpanExporter | None = None
    extra_resource_attributes: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> TelemetryConfig:
        """Return a defensive copy of the telemetry configuration."""
        cloned = replace(self)
        cloned.extra_resource_attributes = dict(self.extra_resource_attributes)
        cloned.openllmetry_config = (
            self.openllmetry_config.clone() if self.openllmetry_config else None
        )
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

    def resolved_openllmetry(self) -> OpenLLMetryConfig | None:
        """Return a normalized OpenLLMetry configuration if enabled."""
        if not self.enable_openllmetry:
            return None

        source = self.openllmetry_config.clone() if self.openllmetry_config else OpenLLMetryConfig()
        source.app_name = source.app_name or self.service_name
        return source
