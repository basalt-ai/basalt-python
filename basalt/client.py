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

        telemetry = TelemetryConfig(
            service_name="my-app",
            environment="production",
            enable_llm_instrumentation=True,
            llm_trace_content=False,
        )
        basalt = Basalt(api_key="your-api-key", telemetry_config=telemetry)
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
        cache : MemoryCache | None = None,
    ):
        """
        Initialize the Basalt client.

        Args:
            api_key: The Basalt API key for authentication.
            telemetry_config: Optional telemetry configuration for OpenTelemetry/OpenLLMetry.
            enable_telemetry: Convenience flag to quickly disable all telemetry.
            base_url: Optional base URL for the API (defaults to config value).
            observability_metadata: Arbitrary metadata dictionary applied to new traces.
        """
        self._api_key = api_key
        self._base_url = base_url

        if not enable_telemetry:
            telemetry_config = TelemetryConfig(enabled=False)
        elif telemetry_config is None:
            telemetry_config = TelemetryConfig()

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
        )
        self._datasets_client = DatasetsClient(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
        )
        self._experiments_client = ExperimentsClient(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
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
