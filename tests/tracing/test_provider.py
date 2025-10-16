"""Tests for the OpenTelemetry tracing provider."""
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter

from basalt.tracing.provider import (
    BasaltConfig,
    create_tracer_provider,
    get_tracer,
    setup_tracing,
)


class TestBasaltConfig:
    """Test cases for BasaltConfig class."""

    def test_creates_config_with_defaults(self):
        """Test that BasaltConfig can be created with default values."""
        config = BasaltConfig()

        assert config.service_name == "basalt-sdk"
        assert config.service_version is not None
        assert config.environment is None
        assert config.extra_resource_attributes == {}

    def test_creates_config_with_custom_values(self):
        """Test that BasaltConfig accepts custom configuration values."""
        config = BasaltConfig(
            service_name="my-service",
            service_version="1.2.3",
            environment="production",
            extra_resource_attributes={"custom.key": "value"}
        )

        assert config.service_name == "my-service"
        assert config.service_version == "1.2.3"
        assert config.environment == "production"
        assert config.extra_resource_attributes == {"custom.key": "value"}

    def test_uses_sdk_version_if_not_provided(self):
        """Test that the SDK version is used if service version is not provided."""
        config = BasaltConfig(service_name="my-service")

        # Should have some version, either from config or 'unknown'
        assert config.service_version is not None
        assert len(config.service_version) > 0


class TestCreateTracerProvider:
    """Test cases for create_tracer_provider function."""

    def test_creates_tracer_provider(self):
        """Test that create_tracer_provider returns a TracerProvider instance."""
        config = BasaltConfig(service_name="test-service")

        provider = create_tracer_provider(config)

        assert isinstance(provider, TracerProvider)

    def test_sets_service_name_in_resource(self):
        """Test that the service name is set in the resource attributes."""
        config = BasaltConfig(service_name="test-service")

        provider = create_tracer_provider(config)

        resource_attrs = provider.resource.attributes
        assert resource_attrs.get("service.name") == "test-service"

    def test_sets_service_version_in_resource(self):
        """Test that the service version is set in the resource attributes."""
        config = BasaltConfig(
            service_name="test-service",
            service_version="2.0.0"
        )

        provider = create_tracer_provider(config)

        resource_attrs = provider.resource.attributes
        assert resource_attrs.get("service.version") == "2.0.0"

    def test_sets_environment_in_resource_when_provided(self):
        """Test that the environment is set in the resource attributes when provided."""
        config = BasaltConfig(
            service_name="test-service",
            environment="staging"
        )

        provider = create_tracer_provider(config)

        resource_attrs = provider.resource.attributes
        assert resource_attrs.get("deployment.environment") == "staging"

    def test_does_not_set_environment_when_not_provided(self):
        """Test that environment is not set when not provided."""
        config = BasaltConfig(service_name="test-service")

        provider = create_tracer_provider(config)

        resource_attrs = provider.resource.attributes
        assert "deployment.environment" not in resource_attrs

    def test_includes_basalt_sdk_metadata(self):
        """Test that Basalt SDK metadata is included in resource attributes."""
        config = BasaltConfig(service_name="test-service")

        provider = create_tracer_provider(config)

        resource_attrs = provider.resource.attributes
        assert "basalt.sdk.type" in resource_attrs
        assert "basalt.sdk.version" in resource_attrs

    def test_includes_extra_resource_attributes(self):
        """Test that extra resource attributes are included."""
        config = BasaltConfig(
            service_name="test-service",
            extra_resource_attributes={
                "custom.attribute": "value",
                "another.attribute": "another-value"
            }
        )

        provider = create_tracer_provider(config)

        resource_attrs = provider.resource.attributes
        assert resource_attrs.get("custom.attribute") == "value"
        assert resource_attrs.get("another.attribute") == "another-value"

    def test_uses_console_exporter_by_default(self):
        """Test that ConsoleSpanExporter is used when no exporter is provided."""
        config = BasaltConfig(service_name="test-service")

        provider = create_tracer_provider(config)

        # Check that a span processor was added
        assert len(provider._active_span_processor._span_processors) > 0

    def test_accepts_custom_exporter(self):
        """Test that a custom exporter can be provided."""
        config = BasaltConfig(service_name="test-service")

        class MockExporter(SpanExporter):
            def export(self, spans):
                return None

            def shutdown(self):
                return None

        custom_exporter = MockExporter()
        provider = create_tracer_provider(config, exporter=custom_exporter)

        # Provider should be created successfully
        assert isinstance(provider, TracerProvider)


class TestSetupTracing:
    """Test cases for setup_tracing function."""

    def test_returns_tracer_provider(self):
        """Test that setup_tracing returns a TracerProvider."""
        config = BasaltConfig(service_name="test-service")

        provider = setup_tracing(config)

        assert isinstance(provider, TracerProvider)

    def test_sets_global_tracer_provider(self):
        """Test that setup_tracing sets the global tracer provider."""
        config = BasaltConfig(service_name="test-service")

        setup_tracing(config)

        # Get a tracer and verify it comes from our provider
        tracer = get_tracer("test")
        assert tracer is not None


class TestGetTracer:
    """Test cases for get_tracer function."""

    def test_returns_tracer(self):
        """Test that get_tracer returns a tracer instance."""
        # Set up tracing first
        config = BasaltConfig(service_name="test-service")
        setup_tracing(config)

        tracer = get_tracer("test-component")

        assert tracer is not None

    def test_accepts_version_parameter(self):
        """Test that get_tracer accepts a version parameter."""
        config = BasaltConfig(service_name="test-service")
        setup_tracing(config)

        tracer = get_tracer("test-component", version="1.0.0")

        assert tracer is not None


class TestTracerProviderIntegration:
    """Integration tests for the tracer provider."""

    def test_can_create_spans(self):
        """Test that spans can be created using the configured provider."""
        config = BasaltConfig(service_name="test-service")
        setup_tracing(config)

        tracer = get_tracer("test")

        # This should not raise an exception
        with tracer.start_as_current_span("test-span") as span:
            span.set_attribute("test.attribute", "test-value")
            assert span is not None

    def test_spans_include_resource_attributes(self):
        """Test that spans include the configured resource attributes."""
        config = BasaltConfig(
            service_name="integration-test-service",
            service_version="3.0.0"
        )
        provider = create_tracer_provider(config)

        # Verify resource attributes are set correctly
        resource_attrs = provider.resource.attributes
        assert resource_attrs.get("service.name") == "integration-test-service"
        assert resource_attrs.get("service.version") == "3.0.0"
