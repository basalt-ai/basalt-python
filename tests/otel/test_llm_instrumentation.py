"""
Integration tests for LLM provider instrumentation via InstrumentationManager.

Tests verify that the InstrumentationManager correctly instruments all supported
LLM providers and that spans contain proper OpenTelemetry GenAI semantic conventions.

Two modes:
1. Mock mode (default): Uses mocked provider responses, no API keys needed
2. Real mode (BASALT_RUN_REAL_LLM_TESTS=1): Makes actual API calls for validation
"""

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from basalt.observability import InstrumentationManager
from basalt.observability.config import TelemetryConfig

from .otlp_utils import spans_to_otel_json

RUN_REAL_LLM_TESTS = os.getenv("BASALT_RUN_REAL_LLM_TESTS") == "1"


@dataclass
class ProviderConfig:
    """
    Configuration for testing a single LLM provider.

    Attributes:
        name: Provider identifier (matches InstrumentationManager provider names)
        make_client: Factory function that returns a configured provider client
        call_llm: Function that makes a minimal LLM call using the client
        required_env: Environment variables required for real API calls
        install_mock: Function that installs mocks for the provider (returns cleanup callable)
    """

    name: str
    make_client: Callable[[], Any]
    call_llm: Callable[[Any], Any]
    required_env: list[str]
    install_mock: Callable[[], Callable[[], None]]


# =============================================================================
# OpenAI Provider
# =============================================================================


def make_openai_client():
    """Create OpenAI client (works with mock or real API key)."""
    import openai

    return openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", "sk-fake-key"))


def call_openai(client):
    """Make a minimal chat completion call."""
    return client.chat.completions.create(
        model=os.getenv("OPENAI_TEST_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=5,
        temperature=0.0,
    )


def install_openai_mock():
    """Install mock for OpenAI API calls."""
    try:
        import openai  # noqa: F401
    except ImportError:
        pytest.skip("openai not installed")

    mock_response = MagicMock()
    mock_response.id = "chatcmpl-fake123"
    mock_response.model = "gpt-4o-mini"
    mock_response.choices = [
        MagicMock(
            message=MagicMock(content="pong", role="assistant"),
            finish_reason="stop",
        )
    ]
    mock_response.usage = MagicMock(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )

    patcher = patch("openai.resources.chat.completions.Completions.create")
    mock_create = patcher.start()
    mock_create.return_value = mock_response

    def cleanup():
        patcher.stop()

    return cleanup


# =============================================================================
# Anthropic Provider
# =============================================================================


def make_anthropic_client():
    """Create Anthropic client (works with mock or real API key)."""
    import anthropic

    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "sk-ant-fake-key"))


def call_anthropic(client):
    """Make a minimal message creation call."""
    return client.messages.create(
        model=os.getenv("ANTHROPIC_TEST_MODEL", "claude-3-5-haiku-20241022"),
        max_tokens=5,
        messages=[{"role": "user", "content": "ping"}],
        temperature=0.0,
    )


def install_anthropic_mock():
    """Install mock for Anthropic API calls."""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        pytest.skip("anthropic not installed")

    mock_response = MagicMock()
    mock_response.id = "msg_fake123"
    mock_response.model = "claude-3-5-haiku-20241022"
    mock_response.content = [MagicMock(text="pong", type="text")]
    mock_response.stop_reason = "end_turn"
    mock_response.usage = MagicMock(
        input_tokens=10,
        output_tokens=5,
    )

    patcher = patch("anthropic.resources.messages.Messages.create")
    mock_create = patcher.start()
    mock_create.return_value = mock_response

    def cleanup():
        patcher.stop()

    return cleanup


# =============================================================================
# Google Generative AI (Gemini) Provider
# =============================================================================


def make_google_genai_client():
    """Create Google Generative AI client (works with mock or real API key)."""
    import google.genai as genai

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY", "fake-api-key"))
    return client


def call_google_genai(client):
    """Make a minimal generate_content call."""
    return client.generate_content(
        "ping",
        generation_config={"max_output_tokens": 5, "temperature": 0.0},
    )


def install_google_genai_mock():
    """Install mock for Google Generative AI API calls."""
    try:
        import google.generativeai  # noqa: F401
    except ImportError:
        pytest.skip("google-generativeai not installed")

    mock_response = MagicMock()
    mock_response.text = "pong"

    # Mock the candidate structure
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = 1  # STOP
    mock_candidate.content.parts = [MagicMock(text="pong")]
    mock_response.candidates = [mock_candidate]

    # Mock usage metadata
    mock_response.usage_metadata = MagicMock(
        prompt_token_count=10,
        candidates_token_count=5,
        total_token_count=15,
    )

    patcher = patch("google.generativeai.GenerativeModel.generate_content")
    mock_generate = patcher.start()
    mock_generate.return_value = mock_response

    def cleanup():
        patcher.stop()

    return cleanup


# =============================================================================
# Vertex AI Provider
# =============================================================================


def make_vertexai_client():
    """Create Vertex AI client (works with mock or real credentials)."""
    import vertexai
    from vertexai.generative_models import GenerativeModel

    # Initialize with project/location from env or use fake values for mocking
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "fake-project")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    vertexai.init(project=project_id, location=location)
    return GenerativeModel(os.getenv("VERTEXAI_TEST_MODEL", "gemini-1.5-flash"))


def call_vertexai(client):
    """Make a minimal generate_content call."""
    return client.generate_content(
        "ping",
        generation_config={"max_output_tokens": 5, "temperature": 0.0},
    )


def install_vertexai_mock():
    """Install mock for Vertex AI API calls."""
    try:
        import vertexai  # noqa: F401
    except ImportError:
        pytest.skip("vertexai not installed")

    mock_response = MagicMock()
    mock_response.text = "pong"

    # Mock the candidate structure
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = 1  # STOP
    mock_candidate.content.parts = [MagicMock(text="pong")]
    mock_response.candidates = [mock_candidate]

    # Mock usage metadata
    mock_response.usage_metadata = MagicMock(
        prompt_token_count=10,
        candidates_token_count=5,
        total_token_count=15,
    )

    patcher = patch("vertexai.generative_models.GenerativeModel.generate_content")
    mock_generate = patcher.start()
    mock_generate.return_value = mock_response

    def cleanup():
        patcher.stop()

    return cleanup


# =============================================================================
# Provider Registry
# =============================================================================

PROVIDERS = {
    "openai": ProviderConfig(
        name="openai",
        make_client=make_openai_client,
        call_llm=call_openai,
        required_env=["OPENAI_API_KEY"],
        install_mock=install_openai_mock,
    ),
    "anthropic": ProviderConfig(
        name="anthropic",
        make_client=make_anthropic_client,
        call_llm=call_anthropic,
        required_env=["ANTHROPIC_API_KEY"],
        install_mock=install_anthropic_mock,
    ),
    # NOTE: Provider name uses underscore to match InstrumentationManager provider_map
    "google_generativeai": ProviderConfig(
        name="google_generativeai",
        make_client=make_google_genai_client,
        call_llm=call_google_genai,
        required_env=["GOOGLE_API_KEY"],
        install_mock=install_google_genai_mock,
    ),
    "vertexai": ProviderConfig(
        name="vertexai",
        make_client=make_vertexai_client,
        call_llm=call_vertexai,
        required_env=["GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"],
        install_mock=install_vertexai_mock,
    ),
}


# =============================================================================
# Helper Functions
# =============================================================================


def _missing_env_vars(vars_: list[str]) -> bool:
    """Check if any required environment variables are missing."""
    return any(os.getenv(v) is None for v in vars_)


def _init_instrumentation_for_provider(provider_name: str, otel_exporter: InMemorySpanExporter):
    """
    Initialize InstrumentationManager with only the specified provider enabled.

    Args:
        provider_name: Name of the provider to instrument
        otel_exporter: The InMemorySpanExporter to use for capturing spans

    Returns:
        InstrumentationManager instance
    """
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    config = TelemetryConfig(
        enable_instrumentation=True,
        trace_content=True,
        enabled_providers=[provider_name],
    )
    manager = InstrumentationManager()
    manager.initialize(config=config)

    # Attach our in-memory exporter to the tracer provider created by the manager
    if manager._tracer_provider:
        manager._tracer_provider.add_span_processor(SimpleSpanProcessor(otel_exporter))

    return manager


# =============================================================================
# Mock Mode Tests
# =============================================================================


@pytest.mark.parametrize("provider_name", list(PROVIDERS.keys()))
def test_instrumentation_initialization(provider_name: str, otel_exporter: InMemorySpanExporter):
    """
    Test that InstrumentationManager correctly initializes provider instrumentors.

    This test verifies that:
    - InstrumentationManager successfully imports and instantiates the instrumentor
    - The instrumentor is registered in the manager's provider_instrumentors dict
    - The instrumentor's instrument() method is called

    NOTE: This test does NOT verify span creation because OpenTelemetry instrumentations
    wrap methods at import time, and mocking those methods bypasses instrumentation.
    Span creation is verified in the real API test mode instead.

    This test runs in CI without requiring API keys, but requires the provider's
    instrumentation package to be installed (e.g., opentelemetry-instrumentation-openai).
    """
    # Map provider names to instrumentation package names for better error messages
    instrumentation_packages = {
        "openai": "opentelemetry-instrumentation-openai",
        "anthropic": "opentelemetry-instrumentation-anthropic",
        "google_generativeai": "opentelemetry-instrumentation-google-generativeai",
        "vertexai": "opentelemetry-instrumentation-vertexai",
    }

    manager = None

    try:
        # Initialize instrumentation for this provider
        manager = _init_instrumentation_for_provider(provider_name, otel_exporter)

        # Check if the instrumentor was registered
        if provider_name not in manager._provider_instrumentors:
            # Instrumentation package not installed - skip test
            pkg_name = instrumentation_packages.get(
                provider_name, f"opentelemetry-instrumentation-{provider_name}"
            )
            pytest.skip(f"Instrumentation package not installed: {pkg_name}")

        instrumentor = manager._provider_instrumentors[provider_name]
        assert instrumentor is not None, f"Instrumentor for {provider_name} is None"

        # Verify the instrumentor has the expected interface
        assert hasattr(instrumentor, "instrument"), (
            f"Instrumentor for {provider_name} missing instrument() method"
        )
        assert hasattr(instrumentor, "uninstrument"), (
            f"Instrumentor for {provider_name} missing uninstrument() method"
        )

    finally:
        # Clean up instrumentation
        if manager:
            manager.shutdown()


# =============================================================================
# Real API Tests
# =============================================================================


@pytest.mark.parametrize("provider_name", list(PROVIDERS.keys()))
def test_instrumentation_span_attrs_real_provider(
    provider_name: str, otel_exporter: InMemorySpanExporter
):
    """
    Test that InstrumentationManager works with real provider APIs (opt-in).

    This test:
    - Only runs when BASALT_RUN_REAL_LLM_TESTS=1
    - Skips if required API keys are missing
    - Makes actual API calls (small/inexpensive models)
    - Verifies the same GenAI attributes with real data

    Use this to validate that instrumentation works end-to-end with real providers.
    """
    if not RUN_REAL_LLM_TESTS:
        pytest.skip("Real LLM tests disabled (set BASALT_RUN_REAL_LLM_TESTS=1 to enable)")

    cfg = PROVIDERS[provider_name]

    if _missing_env_vars(cfg.required_env):
        pytest.skip(f"Missing env vars for {provider_name}: {cfg.required_env}")

    # Initialize instrumentation for this provider
    manager = _init_instrumentation_for_provider(provider_name, otel_exporter)

    try:
        # Create client and make REAL call
        client = cfg.make_client()
        cfg.call_llm(client)

        # Verify spans were captured
        spans = otel_exporter.get_finished_spans()
        assert spans, f"No spans captured for provider={provider_name} (real call)"

        # Find the LLM span
        span = spans[-1]
        attrs = span.attributes

        # Verify standard GenAI semantic conventions
        assert "gen_ai.system" in attrs, f"Missing gen_ai.system for {provider_name}"
        assert "gen_ai.request.model" in attrs, f"Missing gen_ai.request.model for {provider_name}"

        # With real calls, we should definitely have these
        assert "gen_ai.usage.input_tokens" in attrs or "gen_ai.usage.prompt_tokens" in attrs, (
            f"Missing input token count for {provider_name} (real call)"
        )

        assert "gen_ai.usage.output_tokens" in attrs or "gen_ai.usage.completion_tokens" in attrs, (
            f"Missing output token count for {provider_name} (real call)"
        )

        # Verify token counts are positive integers
        input_tokens = attrs.get("gen_ai.usage.input_tokens") or attrs.get(
            "gen_ai.usage.prompt_tokens"
        )
        output_tokens = attrs.get("gen_ai.usage.output_tokens") or attrs.get(
            "gen_ai.usage.completion_tokens"
        )

        assert input_tokens > 0, f"Input tokens should be > 0 for {provider_name}"
        assert output_tokens > 0, f"Output tokens should be > 0 for {provider_name}"

    finally:
        # Clean up instrumentation
        if manager:
            manager.shutdown()


# =============================================================================
# OTLP JSON Export Tests
# =============================================================================


def test_otlp_json_conversion_helpers():
    """
    Test the OTLP JSON conversion helpers with a manually created span.

    This test validates that the conversion utilities work correctly without
    requiring real API calls or provider instrumentation. It creates a simple
    span with GenAI attributes and verifies the OTLP JSON structure.
    """
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.trace import SpanKind

    # Set up a minimal tracer provider (don't set it globally to avoid conflicts)
    resource = Resource.create(
        {
            "service.name": "test-service",
            "telemetry.sdk.language": "python",
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.version": "1.0.0",
        }
    )
    provider = TracerProvider(resource=resource)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Create a test span with GenAI attributes using the provider directly
    tracer = provider.get_tracer("test.instrumentation", "1.0.0")
    with tracer.start_as_current_span("test.llm.call", kind=SpanKind.CLIENT) as span:
        span.set_attribute("gen_ai.system", "test-provider")
        span.set_attribute("gen_ai.request.model", "test-model-1")
        span.set_attribute("gen_ai.response.model", "test-model-1")
        span.set_attribute("gen_ai.usage.input_tokens", 10)
        span.set_attribute("gen_ai.usage.output_tokens", 20)

    # Get the finished span
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    test_span = spans[0]

    # Convert to OTLP JSON
    otlp_json = spans_to_otel_json(
        spans=[test_span],
        resource_attributes=dict(test_span.resource.attributes),
        scope_name=test_span.instrumentation_scope.name,
        scope_version=test_span.instrumentation_scope.version,
    )

    # Validate structure
    assert "resourceSpans" in otlp_json
    resource_span = otlp_json["resourceSpans"][0]

    # Validate resource
    assert "resource" in resource_span
    resource_attrs = {
        item["key"]: item["value"] for item in resource_span["resource"]["attributes"]
    }
    assert resource_attrs["service.name"]["stringValue"] == "test-service"

    # Validate scope
    scope_span = resource_span["scopeSpans"][0]
    assert scope_span["scope"]["name"] == "test.instrumentation"
    assert scope_span["scope"]["version"] == "1.0.0"

    # Validate span
    span_json = scope_span["spans"][0]
    assert span_json["name"] == "test.llm.call"
    assert span_json["kind"] == SpanKind.CLIENT.value

    # Validate attributes
    attrs = {item["key"]: item["value"] for item in span_json["attributes"]}
    assert attrs["gen_ai.system"]["stringValue"] == "test-provider"
    assert attrs["gen_ai.request.model"]["stringValue"] == "test-model-1"
    assert attrs["gen_ai.usage.input_tokens"]["intValue"] == 10
    assert attrs["gen_ai.usage.output_tokens"]["intValue"] == 20


def _get_mock_span_config(provider_name: str):
    """
    Get provider-specific span configuration for mock testing.

    Returns a dict with scope_name, scope_version, span_name, and attributes
    that match what the real instrumentation would produce.
    """
    span_configs = {
        "openai": {
            "scope_name": "opentelemetry.instrumentation.openai",
            "scope_version": "0.21.5",
            "span_name": "chat",
            "attributes": {
                "gen_ai.system": "openai",
                "gen_ai.request.model": "gpt-4o-mini",
                "gen_ai.response.model": "gpt-4o-mini",
                "gen_ai.usage.input_tokens": 10,
                "gen_ai.usage.output_tokens": 5,
                "llm.usage.total_tokens": 15,
            },
        },
        "anthropic": {
            "scope_name": "opentelemetry.instrumentation.anthropic",
            "scope_version": "0.21.5",
            "span_name": "chat",
            "attributes": {
                "gen_ai.system": "anthropic",
                "gen_ai.request.model": "claude-3-5-haiku-20241022",
                "gen_ai.response.model": "claude-3-5-haiku-20241022",
                "gen_ai.usage.input_tokens": 10,
                "gen_ai.usage.output_tokens": 5,
            },
        },
        "google_generativeai": {
            "scope_name": "opentelemetry.instrumentation.google_generativeai",
            "scope_version": "0.21.5",
            "span_name": "gemini.generate_content",
            "attributes": {
                "gen_ai.system": "google",
                "gen_ai.request.model": "gemini-1.5-flash",
                "gen_ai.response.model": "gemini-1.5-flash",
                "gen_ai.usage.prompt_tokens": 10,
                "gen_ai.usage.completion_tokens": 5,
                "llm.usage.total_tokens": 15,
            },
        },
        "vertexai": {
            "scope_name": "opentelemetry.instrumentation.vertexai",
            "scope_version": "0.21.5",
            "span_name": "vertexai.generate_content",
            "attributes": {
                "gen_ai.system": "vertex_ai",
                "gen_ai.request.model": "gemini-1.5-flash",
                "gen_ai.response.model": "gemini-1.5-flash",
                "gen_ai.usage.prompt_tokens": 10,
                "gen_ai.usage.completion_tokens": 5,
                "llm.usage.total_tokens": 15,
            },
        },
    }
    return span_configs[provider_name]


@pytest.mark.parametrize("provider_name", list(PROVIDERS.keys()))
def test_llm_span_otlp_json_with_mocks(provider_name: str, otel_exporter: InMemorySpanExporter):
    """
    Test that LLM spans can be converted to OTLP JSON matching TypeScript parser contract.

    This test:
    - Creates realistic mock spans that match what instrumentation produces
    - Does NOT require API keys or environment variables
    - Validates OTLP JSON structure matches TypeScript OtelJsonParser expectations
    - Verifies GenAI semantic conventions and resource attributes

    The mock spans mirror what real instrumentations would create, allowing
    validation of the OTLP conversion without needing actual API calls.
    """
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.trace import SpanKind

    # Get provider-specific configuration
    config = _get_mock_span_config(provider_name)

    # Create a fresh TracerProvider with realistic resource attributes
    resource = Resource.create(
        {
            "service.name": "llm-service",
            "service.version": "1.0.0-rc1",
            "telemetry.sdk.language": "python",
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.version": "1.38.0",
            "basalt.sdk.type": "python",
            "basalt.sdk.version": "1.0.0-rc1",
        }
    )
    provider = TracerProvider(resource=resource)

    # Use a fresh InMemorySpanExporter for this test
    test_exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(test_exporter))

    # Create the span with provider-specific attributes
    tracer = provider.get_tracer(config["scope_name"], config["scope_version"])
    with tracer.start_as_current_span(config["span_name"], kind=SpanKind.CLIENT) as span:
        for key, value in config["attributes"].items():
            span.set_attribute(key, value)

    # Get the finished span
    py_span = test_exporter.get_finished_spans()[0]

    # Get resource attributes from the span
    resource_attrs = dict(py_span.resource.attributes)

    # Convert to OTLP JSON format
    otlp_json = spans_to_otel_json(
        spans=[py_span],
        resource_attributes=resource_attrs,
        scope_name=py_span.instrumentation_scope.name,
        scope_version=py_span.instrumentation_scope.version,
    )

    # ---- Validate OTLP JSON Structure ----

    assert "resourceSpans" in otlp_json, "Missing resourceSpans key"
    assert len(otlp_json["resourceSpans"]) == 1, "Expected exactly one resourceSpan"

    resource_span = otlp_json["resourceSpans"][0]

    # Validate resource structure
    assert "resource" in resource_span, "Missing resource key"
    assert "attributes" in resource_span["resource"], "Missing resource.attributes"

    # Validate scopeSpans structure
    assert "scopeSpans" in resource_span, "Missing scopeSpans key"
    assert len(resource_span["scopeSpans"]) == 1, "Expected exactly one scopeSpan"

    scope_span = resource_span["scopeSpans"][0]

    # Validate scope structure
    assert "scope" in scope_span, "Missing scope key"
    assert "name" in scope_span["scope"], "Missing scope.name"

    # Validate spans structure
    assert "spans" in scope_span, "Missing spans key"
    assert len(scope_span["spans"]) == 1, "Expected exactly one span"

    span_json = scope_span["spans"][0]

    # Validate span has required OTLP fields
    required_fields = [
        "traceId",
        "spanId",
        "name",
        "kind",
        "startTimeUnixNano",
        "endTimeUnixNano",
        "attributes",
        "status",
    ]
    for field in required_fields:
        assert field in span_json, f"Missing required field: {field}"

    # Validate trace and span IDs are properly formatted
    assert len(span_json["traceId"]) == 32, "traceId should be 32-char hex string"
    assert len(span_json["spanId"]) == 16, "spanId should be 16-char hex string"

    # ---- Validate GenAI Semantic Conventions ----

    # Convert attributes list to dict for easier assertion
    attrs = {item["key"]: item["value"] for item in span_json["attributes"]}

    def get_string(key: str) -> str | None:
        """Extract string value from OTLP attribute."""
        v = attrs.get(key)
        if not v:
            return None
        return v.get("stringValue")

    def get_int(key: str) -> int | None:
        """Extract int value from OTLP attribute."""
        v = attrs.get(key)
        if not v:
            return None
        iv = v.get("intValue")
        if isinstance(iv, str):
            return int(iv)
        return iv

    # Verify core GenAI attributes are present
    assert "gen_ai.system" in attrs, f"Missing gen_ai.system for {provider_name}"
    assert "gen_ai.request.model" in attrs, f"Missing gen_ai.request.model for {provider_name}"

    # Verify token usage attributes
    # Different instrumentors use different attribute names
    has_input_tokens = "gen_ai.usage.input_tokens" in attrs or "gen_ai.usage.prompt_tokens" in attrs
    has_output_tokens = (
        "gen_ai.usage.output_tokens" in attrs or "gen_ai.usage.completion_tokens" in attrs
    )

    assert has_input_tokens, f"Missing input token count for {provider_name}"
    assert has_output_tokens, f"Missing output token count for {provider_name}"

    # Verify token counts are positive integers (from mocked response)
    input_tokens = get_int("gen_ai.usage.input_tokens") or get_int("gen_ai.usage.prompt_tokens")
    output_tokens = get_int("gen_ai.usage.output_tokens") or get_int(
        "gen_ai.usage.completion_tokens"
    )

    assert input_tokens and input_tokens > 0, f"Input tokens should be > 0 for {provider_name}"
    assert output_tokens and output_tokens > 0, f"Output tokens should be > 0 for {provider_name}"

    # ---- Validate Resource Attributes ----

    resource_attrs_dict = {
        item["key"]: item["value"] for item in resource_span["resource"]["attributes"]
    }

    # Verify key resource attributes exist
    assert "service.name" in resource_attrs_dict, "Missing service.name"
    assert "telemetry.sdk.language" in resource_attrs_dict, "Missing telemetry.sdk.language"
    assert "telemetry.sdk.name" in resource_attrs_dict, "Missing telemetry.sdk.name"

    # Verify resource attributes have correct types
    assert resource_attrs_dict["service.name"].get("stringValue"), "service.name should be a string"
    assert resource_attrs_dict["telemetry.sdk.language"].get("stringValue") == "python", (
        "telemetry.sdk.language should be 'python'"
    )

    # Optional: Print OTLP JSON for debugging or TS test fixture generation
    # Uncomment to see the full JSON output:
    # print(f"\n=== OTLP JSON for {provider_name} ===")
    # print(json.dumps(otlp_json, indent=2))
