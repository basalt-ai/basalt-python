"""
Main Basalt SDK client.

This module provides the main Basalt client class for interacting with the Basalt API.
"""
from __future__ import annotations

from opentelemetry.sdk.trace.export import SpanExporter

from basalt._internal.http import HTTPClient

from .datasets.client import DatasetsClient
from .instrumentation.openai import OpenAIInstrumentor
from .prompts.client import PromptsClient
from .tracing.provider import BasaltConfig, setup_tracing
from .utils.memcache import MemoryCache


class Basalt:
    """
    Main client for the Basalt SDK.

    This client provides access to the Basalt API services including prompts and datasets,
    with built-in tracing support via OpenTelemetry.

    Example:
        ```python
        from basalt import Basalt
        from basalt.tracing.provider import BasaltConfig

        # Initialize the client
        config = BasaltConfig(
            service_name="my-app",
            service_version="1.0.0",
            environment="production"
        )
        basalt = Basalt(
            api_key="your-api-key",
            config=config
        )

        # Use the prompts client
        prompt = await basalt.prompts.get("my-prompt")

        # Use the datasets client
        datasets = await basalt.datasets.list()
        ```
    """

    def __init__(
        self,
        api_key: str,
        config: BasaltConfig | None = None,
        base_url: str | None = None,
        exporter: SpanExporter | None = None,
        instrument_openai: bool = True,
    ):
        """
        Initialize the Basalt client.

        Args:
            api_key: The Basalt API key for authentication.
            config: Optional tracing configuration. If not provided, default config is used.
            base_url: Optional base URL for the API (defaults to config value).
            exporter: Optional OpenTelemetry SpanExporter for custom tracing backends.
            instrument_openai: Whether to automatically instrument the OpenAI SDK (default: True).
        """
        self._api_key = api_key
        self._base_url = base_url

        # Initialize tracing
        self._config = config or BasaltConfig()
        self._tracer_provider = setup_tracing(self._config, exporter)

        # Initialize OpenAI instrumentation
        self._openai_instrumentor = None
        if instrument_openai:
            self._openai_instrumentor = OpenAIInstrumentor(tracer_provider=self._tracer_provider)
            self._openai_instrumentor.instrument()

        # Initialize caches
        self._cache = MemoryCache()
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

    def shutdown(self):
        """
        Shutdown the client and flush any pending telemetry data.

        This ensures all spans are exported before the application exits.
        """
        from .tracing.provider import shutdown_tracing
        shutdown_tracing()
