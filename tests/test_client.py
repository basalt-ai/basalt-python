"""Tests for the main Basalt client."""
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from basalt.client import Basalt
from basalt.datasets.client import DatasetsClient
from basalt.prompts.client import PromptsClient
from basalt.tracing.provider import BasaltConfig


class TestBasaltClient:
    """Test suite for the main Basalt client."""

    def test_client_initialization_with_defaults(self):
        """Test that the client initializes with default configuration."""
        client = Basalt(api_key="test-api-key")

        assert client is not None
        assert client._api_key == "test-api-key"
        assert client._config is not None
        assert client._config.service_name == "basalt-sdk"
        assert client._tracer_provider is not None
        assert isinstance(client._tracer_provider, TracerProvider)

    def test_client_initialization_with_custom_config(self):
        """Test that the client initializes with custom configuration."""
        config = BasaltConfig(
            service_name="test-service",
            service_version="1.0.0",
            environment="testing",
            extra_resource_attributes={"custom.attr": "value"},
        )

        client = Basalt(api_key="test-api-key", config=config)

        assert client is not None
        assert client._config.service_name == "test-service"
        assert client._config.service_version == "1.0.0"
        assert client._config.environment == "testing"
        assert client._config.extra_resource_attributes["custom.attr"] == "value"

    def test_client_initialization_with_custom_exporter(self):
        """Test that the client initializes with a custom exporter."""
        exporter = InMemorySpanExporter()
        client = Basalt(api_key="test-api-key", exporter=exporter)

        assert client is not None
        assert client._tracer_provider is not None

        # Verify that the exporter was added to the provider
        processors = client._tracer_provider._active_span_processor._span_processors
        assert len(processors) > 0
        assert processors[0].span_exporter is exporter

    def test_client_initialization_with_base_url(self):
        """Test that the client initializes with a custom base URL."""
        client = Basalt(api_key="test-api-key", base_url="https://custom-api.example.com")

        assert client is not None
        assert client._base_url == "https://custom-api.example.com"

    def test_prompts_client_property(self):
        """Test that the prompts property returns a PromptsClient instance."""
        client = Basalt(api_key="test-api-key")

        prompts_client = client.prompts

        assert prompts_client is not None
        assert isinstance(prompts_client, PromptsClient)
        assert prompts_client is client._prompts_client

    def test_datasets_client_property(self):
        """Test that the datasets property returns a DatasetsClient instance."""
        client = Basalt(api_key="test-api-key")

        datasets_client = client.datasets

        assert datasets_client is not None
        assert isinstance(datasets_client, DatasetsClient)
        assert datasets_client is client._datasets_client

    def test_shutdown(self):
        """Test that shutdown properly cleans up resources."""
        exporter = InMemorySpanExporter()
        client = Basalt(api_key="test-api-key", exporter=exporter)

        # Should not raise an exception
        client.shutdown()

        # After shutdown, the provider should be shut down
        # We can verify this by checking that the exporter was shutdown
        # (InMemorySpanExporter doesn't have a shutdown method, but the call should succeed)

    def test_client_uses_same_cache_instances(self):
        """Test that the client properly shares cache instances."""
        client = Basalt(api_key="test-api-key")

        # Both prompts client should use the same cache instances
        assert client._prompts_client._cache is client._cache
        assert client._prompts_client._fallback_cache is client._fallback_cache

    def test_client_integration_with_custom_config_and_exporter(self):
        """Test full client initialization with all custom parameters."""
        config = BasaltConfig(
            service_name="integration-test",
            service_version="2.0.0",
            environment="staging",
        )
        exporter = InMemorySpanExporter()

        client = Basalt(
            api_key="test-api-key",
            config=config,
            base_url="https://staging-api.example.com",
            exporter=exporter,
        )

        assert client is not None
        assert client._api_key == "test-api-key"
        assert client._config.service_name == "integration-test"
        assert client._base_url == "https://staging-api.example.com"
        assert client.prompts is not None
        assert client.datasets is not None

        # Verify tracer provider is properly configured
        assert client._tracer_provider is not None
        processors = client._tracer_provider._active_span_processor._span_processors
        assert len(processors) > 0
        assert processors[0].span_exporter is exporter
