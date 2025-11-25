"""
Main Basalt SDK client.

This module provides the main Basalt client class for interacting with the Basalt API.
"""
from __future__ import annotations

from typing import Any

from basalt._internal.http import HTTPClient
from basalt.observability.config import TelemetryConfig
from basalt.observability.instrumentation import InstrumentationManager
from basalt.observability.trace_context import configure_global_metadata
from basalt.types.cache import CacheProtocol

from .datasets.client import DatasetsClient
from .experiments.client import ExperimentsClient
from .prompts.client import PromptsClient
from .utils.memcache import MemoryCache


class Basalt:
    """
    Main client for the Basalt SDK.

    This client provides access to the Basalt API services including prompts and datasets,
    with built-in tracing support via OpenTelemetry.

    Example:
        ```python
        from basalt import Basalt, TelemetryConfig

        # Using TelemetryConfig for advanced configuration
        telemetry = TelemetryConfig(
            service_name="my-app",
            environment="production",
            enable_instrumentation=True,
            trace_content=False,
        )
        basalt = Basalt(api_key="your-api-key", telemetry_config=telemetry)

        # Or use client-level parameters for simple cases
        basalt = Basalt(
            api_key="your-api-key",
            enabled_instruments=["openai", "anthropic"]
        )
        ```
    """

    def __init__(
        self,
        api_key: str,
        *,
        telemetry_config: TelemetryConfig | None = None,
        enable_telemetry: bool = True,
        base_url: str | None = None,
        observability_metadata: dict[str, Any] | None = None,
        cache : CacheProtocol | None = None,
        log_level: str | None = None,
        enabled_instruments: list[str] | None = None,
        disabled_instruments: list[str] | None = None,
    ):
        """
        Initialize the Basalt client.

        Args:
            api_key: The Basalt API key for authentication.
            telemetry_config: Optional telemetry configuration for OpenTelemetry/OpenLLMetry.
            enable_telemetry: Convenience flag to quickly disable all telemetry.
            base_url: Optional base URL for the API (defaults to config value).
            observability_metadata: Arbitrary metadata dictionary applied to new traces.
            log_level: Optional log level for API client loggers (e.g., 'DEBUG', 'INFO', 'WARNING').
            enabled_instruments: List of specific instruments to enable (e.g., ["openai", "anthropic"]).
                Takes precedence over telemetry_config if provided.
            disabled_instruments: List of instruments to explicitly disable (e.g., ["langchain"]).
                Takes precedence over telemetry_config if provided.
        """
        self._api_key = api_key
        self._base_url = base_url

        if not enable_telemetry:
            telemetry_config = TelemetryConfig(enabled=False)
        elif telemetry_config is None:
            telemetry_config = TelemetryConfig()

        # Apply client-level instrument settings if provided (takes precedence)
        if enabled_instruments is not None or disabled_instruments is not None:
            telemetry_config = telemetry_config.clone()
            if enabled_instruments is not None:
                telemetry_config.enabled_providers = enabled_instruments
            if disabled_instruments is not None:
                telemetry_config.disabled_providers = disabled_instruments

        self._telemetry_config = telemetry_config
        self._instrumentation = InstrumentationManager()
        self._instrumentation.initialize(telemetry_config, api_key=api_key)

        # Configure global observability metadata if provided
        if observability_metadata:
            configure_global_metadata(observability_metadata)

        # Initialize caches
        self._cache = cache or MemoryCache()
        self._fallback_cache = MemoryCache()

        http_client = HTTPClient()

        # Initialize sub-clients
        self._prompts_client = PromptsClient(
            api_key=api_key,
            cache=self._cache,
            fallback_cache=self._fallback_cache,
            base_url=base_url,
            http_client=http_client,
            log_level=log_level,
        )
        self._datasets_client = DatasetsClient(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
            log_level=log_level,
        )
        self._experiments_client = ExperimentsClient(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
            log_level=log_level,
        )

    @property
    def prompts(self) -> PromptsClient:
        """
        Access the Prompts API client.

        Returns:
            PromptsClient instance for interacting with prompts.
        """
        return self._prompts_client

    @property
    def datasets(self) -> DatasetsClient:
        """
        Access the Datasets API client.

        Returns:
            DatasetsClient instance for interacting with datasets.
        """
        return self._datasets_client

    @property
    def experiments(self) -> ExperimentsClient:
        """
        Access the Experiments API client.

        Returns:
            ExperimentsClient instance for interacting with experiments/features.
        """
        return self._experiments_client

    def shutdown(self):
        """
        Shutdown the client and flush any pending telemetry data.

        This ensures all spans are exported before the application exits.
        """
        self._instrumentation.shutdown()
