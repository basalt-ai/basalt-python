"""Tests for OpenAI instrumentation."""
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from basalt.client import Basalt
from basalt.instrumentation.openai import OpenAIInstrumentor
from basalt.tracing.provider import BasaltConfig


@pytest.fixture(scope="session")
def session_tracer_provider_for_openai():
    """Set up a global tracer provider for all tests in this module."""
    from opentelemetry import trace

    exporter = InMemorySpanExporter()
    config = BasaltConfig(service_name="test-openai-instrumentation")

    from basalt.tracing.provider import create_tracer_provider
    provider = create_tracer_provider(config, exporter)
    trace._set_tracer_provider(provider, log=False)

    yield provider, exporter

    # Cleanup after all tests
    trace._TRACER_PROVIDER = None


@pytest.fixture
def in_memory_exporter_for_openai(session_tracer_provider_for_openai):
    """Fixture that provides an InMemorySpanExporter for capturing spans."""
    provider, exporter = session_tracer_provider_for_openai

    # Clear spans before each test to start fresh
    exporter.clear()

    yield exporter

    # Clear spans after each test
    exporter.clear()


class TestOpenAIInstrumentor:
    """Test suite for OpenAI instrumentation."""

    def test_instrumentor_initialization(self):
        """Test that the instrumentor can be initialized."""
        instrumentor = OpenAIInstrumentor()
        assert instrumentor is not None
        assert not instrumentor.is_instrumented

    def test_instrumentor_without_openai_installed(self, in_memory_exporter_for_openai):
        """Test that instrumentation gracefully handles missing OpenAI."""
        instrumentor = OpenAIInstrumentor()

        # Mock the import to raise ImportError
        with patch.dict(sys.modules, {"openai": None}):
            # Should not raise an error
            instrumentor.instrument()

        # Should not be marked as instrumented
        assert not instrumentor.is_instrumented

    @patch("basalt.instrumentation.openai.wrapt")
    def test_instrumentor_patches_sync_completions(self, mock_wrapt, in_memory_exporter_for_openai):
        """Test that the instrumentor patches sync chat completions."""
        # Create a mock openai module
        mock_openai = MagicMock()
        mock_openai.OpenAI = MagicMock()

        with patch.dict(sys.modules, {"openai": mock_openai}):
            instrumentor = OpenAIInstrumentor()
            instrumentor.instrument()

            # Verify wrapt.wrap_function_wrapper was called
            assert mock_wrapt.wrap_function_wrapper.called
            calls = mock_wrapt.wrap_function_wrapper.call_args_list

            # Check that we wrapped the sync completion method
            sync_call = [c for c in calls if "Completions.create" in str(c)]
            assert len(sync_call) > 0

    @patch("basalt.instrumentation.openai.wrapt")
    def test_instrumentor_patches_async_completions(self, mock_wrapt, in_memory_exporter_for_openai):
        """Test that the instrumentor patches async chat completions."""
        # Create a mock openai module
        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI = MagicMock()

        with patch.dict(sys.modules, {"openai": mock_openai}):
            instrumentor = OpenAIInstrumentor()
            instrumentor.instrument()

            # Verify wrapt.wrap_function_wrapper was called
            assert mock_wrapt.wrap_function_wrapper.called
            calls = mock_wrapt.wrap_function_wrapper.call_args_list

            # Check that we wrapped the async completion method
            async_call = [c for c in calls if "AsyncCompletions.create" in str(c)]
            assert len(async_call) > 0

    def test_instrumentor_double_instrument_warning(self, in_memory_exporter_for_openai):
        """Test that instrumenting twice logs a warning."""
        # Create a mock openai module
        mock_openai = MagicMock()
        mock_openai.OpenAI = MagicMock()

        with patch.dict(sys.modules, {"openai": mock_openai}):
            instrumentor = OpenAIInstrumentor()
            instrumentor.instrument()

            # Mark as instrumented manually
            instrumentor._is_instrumented = True

            # Try to instrument again - should log warning
            with patch("basalt.instrumentation.openai.logger") as mock_logger:
                instrumentor.instrument()
                mock_logger.warning.assert_called_once()

    def test_extract_response_attributes(self, in_memory_exporter_for_openai):
        """Test extraction of response attributes."""
        instrumentor = OpenAIInstrumentor()

        # Create a mock response
        mock_response = Mock()
        mock_response.id = "chatcmpl-123"
        mock_response.model = "gpt-4"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.choices = [Mock()]
        mock_response.choices[0].finish_reason = "stop"

        attrs = instrumentor._extract_response_attributes(mock_response)

        assert attrs["gen_ai.response.id"] == "chatcmpl-123"
        assert attrs["gen_ai.response.model"] == "gpt-4"
        assert attrs["gen_ai.usage.input_tokens"] == 10
        assert attrs["gen_ai.usage.output_tokens"] == 20
        assert attrs["gen_ai.response.finish_reasons"] == ["stop"]

    def test_extract_response_attributes_handles_missing_fields(self, in_memory_exporter_for_openai):
        """Test that extraction handles missing fields gracefully."""
        instrumentor = OpenAIInstrumentor()

        # Create a mock response with minimal fields using spec to prevent auto-attributes
        mock_response = Mock(spec=["id"])
        mock_response.id = "chatcmpl-123"

        attrs = instrumentor._extract_response_attributes(mock_response)

        # Should have at least the ID
        assert attrs["gen_ai.response.id"] == "chatcmpl-123"
        # Should not crash on missing fields
        assert "gen_ai.response.model" not in attrs
        assert "gen_ai.usage.input_tokens" not in attrs


class TestBasaltClientWithInstrumentation:
    """Test suite for Basalt client with OpenAI instrumentation."""

    def test_basalt_client_auto_instruments_openai(self, in_memory_exporter_for_openai):
        """Test that initializing Basalt client auto-instruments OpenAI."""
        # Create a mock openai module
        mock_openai = MagicMock()
        mock_openai.OpenAI = MagicMock()

        with patch.dict(sys.modules, {"openai": mock_openai}):
            with patch("basalt.instrumentation.openai.wrapt") as mock_wrapt:
                client = Basalt(api_key="test-key")

                # Verify that OpenAI was instrumented
                assert client._openai_instrumentor is not None
                assert mock_wrapt.wrap_function_wrapper.called

    def test_basalt_client_respects_instrument_openai_flag(self, in_memory_exporter_for_openai):
        """Test that instrument_openai=False skips instrumentation."""
        client = Basalt(api_key="test-key", instrument_openai=False)

        # Verify that OpenAI instrumentor was not created
        assert client._openai_instrumentor is None

    @pytest.mark.asyncio
    async def test_instrumented_openai_creates_spans(self, in_memory_exporter_for_openai):
        """Test that instrumented OpenAI calls create spans."""
        # This is an integration test that requires a more complex mock setup
        # For now, we'll test the wrapper function directly

        from basalt.instrumentation.openai import OpenAIInstrumentor

        instrumentor = OpenAIInstrumentor()

        # Create a mock wrapped function
        def mock_create(*args, **kwargs):
            # Create a mock response
            mock_response = Mock()
            mock_response.id = "chatcmpl-test"
            mock_response.model = "gpt-4"
            mock_response.usage = Mock()
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 20
            mock_response.choices = [Mock()]
            mock_response.choices[0].finish_reason = "stop"
            return mock_response

        # Call the wrapper
        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "max_tokens": 100,
        }

        result = instrumentor._wrap_chat_completion(
            wrapped=mock_create,
            instance=None,
            args=(),
            kwargs=kwargs,
        )

        # Verify response is returned
        assert result is not None
        assert result.id == "chatcmpl-test"

        # Verify span was created
        spans = in_memory_exporter_for_openai.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "openai.chat.completions.create"

        # Verify request attributes
        attrs = dict(span.attributes or {})
        assert attrs.get("gen_ai.system") == "openai"
        assert attrs.get("gen_ai.request.model") == "gpt-4"
        assert attrs.get("gen_ai.operation.name") == "chat.completions"
        assert attrs.get("gen_ai.request.temperature") == 0.7
        assert attrs.get("gen_ai.request.max_tokens") == 100

        # Verify response attributes
        assert attrs.get("gen_ai.response.id") == "chatcmpl-test"
        assert attrs.get("gen_ai.response.model") == "gpt-4"
        assert attrs.get("gen_ai.usage.input_tokens") == 10
        assert attrs.get("gen_ai.usage.output_tokens") == 20

    def test_instrumented_openai_handles_exceptions(self, in_memory_exporter_for_openai):
        """Test that instrumented OpenAI calls handle exceptions properly."""
        from opentelemetry.trace import StatusCode

        from basalt.instrumentation.openai import OpenAIInstrumentor

        instrumentor = OpenAIInstrumentor()

        # Create a mock wrapped function that raises an exception
        def mock_create_with_error(*args, **kwargs):
            raise ValueError("API Error")

        # Call the wrapper
        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        with pytest.raises(ValueError, match="API Error"):
            instrumentor._wrap_chat_completion(
                wrapped=mock_create_with_error,
                instance=None,
                args=(),
                kwargs=kwargs,
            )

        # Verify span was created and marked as error
        spans = in_memory_exporter_for_openai.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR

        # Verify exception was recorded
        events = span.events
        assert len(events) == 1
        assert events[0].name == "exception"
