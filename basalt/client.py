"""
Main Basalt SDK client.

This module provides the main Basalt client class for interacting with the Basalt API.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from basalt._internal.http import HTTPClient
from basalt.observability import configure_trace_defaults
from basalt.observability.config import TelemetryConfig
from basalt.observability.instrumentation import InstrumentationManager
from basalt.observability.trace_context import (
    TraceContextConfig,
    TraceExperiment,
    TraceIdentity,
)

from .datasets.client import DatasetsClient
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
        trace_context: TraceContextConfig | None = None,
        trace_user: TraceIdentity | dict[str, str] | None = None,
        trace_organization: TraceIdentity | dict[str, str] | None = None,
        trace_experiment: TraceExperiment | dict[str, Any] | None = None,
        trace_metadata: dict[str, Any] | None = None,
        trace_evaluators: Iterable[str] | None = None,
    ):
        """
        Initialize the Basalt client.

        Args:
            api_key: The Basalt API key for authentication.
            telemetry_config: Optional telemetry configuration for OpenTelemetry/OpenLLMetry.
            enable_telemetry: Convenience flag to quickly disable all telemetry.
            base_url: Optional base URL for the API (defaults to config value).
            trace_context: Optional trace defaults applied to every span created via the SDK.
            trace_user: Convenience shortcut to set the default trace user (overrides trace_context).
            trace_organization: Convenience shortcut to set the default trace organization.
            trace_experiment: Default experiment metadata to attach to traces.
            trace_metadata: Arbitrary metadata dictionary applied to new traces.
            trace_evaluators: Iterable of evaluator slugs attached to spans by default.
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

        context_payload: dict[str, Any] = {}
        if trace_context is not None:
            if trace_context.user is not None:
                context_payload["user"] = trace_context.user
            if trace_context.organization is not None:
                context_payload["organization"] = trace_context.organization
            if trace_context.experiment is not None:
                context_payload["experiment"] = trace_context.experiment
            if trace_context.metadata is not None:
                context_payload["metadata"] = dict(trace_context.metadata)
            if trace_context.evaluators is not None:
                context_payload["evaluators"] = list(trace_context.evaluators)

        if trace_user is not None:
            context_payload["user"] = trace_user
        if trace_organization is not None:
            context_payload["organization"] = trace_organization
        if trace_experiment is not None:
            context_payload["experiment"] = trace_experiment
        if trace_metadata is not None:
            context_payload["metadata"] = trace_metadata
        if trace_evaluators is not None:
            context_payload["evaluators"] = list(trace_evaluators)

        if context_payload:
            configure_trace_defaults(**context_payload)

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
        self._instrumentation.shutdown()
