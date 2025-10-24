"""Tests for tracing decorators."""
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from basalt.tracing.decorators import trace, trace_function
from basalt.tracing.provider import BasaltConfig


@pytest.fixture(scope="session")
def session_tracer_provider():
    """Set up a global tracer provider for all tests in this module."""
    from opentelemetry import trace

    exporter = InMemorySpanExporter()
    config = BasaltConfig(service_name="test-service")

    from basalt.tracing.provider import create_tracer_provider
    provider = create_tracer_provider(config, exporter)
    # Force reset any existing global provider to ensure isolation for this module
    trace._TRACER_PROVIDER = None
    trace._set_tracer_provider(provider, log=False)

    yield provider, exporter

    # Cleanup after all tests
    trace._TRACER_PROVIDER = None


@pytest.fixture(autouse=False)
def in_memory_exporter(session_tracer_provider):
    """Fixture that provides an InMemorySpanExporter for capturing spans."""
    provider, exporter = session_tracer_provider

    # Clear spans before each test to start fresh
    exporter.clear()

    yield exporter

    # Clear spans after each test
    exporter.clear()


class TestTraceDecorator:
    """Test suite for the @trace decorator."""

    def test_trace_sync_function_creates_span(self, in_memory_exporter):
        """Test that the @trace decorator creates a span for sync functions."""
        @trace()
        def sample_function(x: int) -> int:
            return x * 2

        result = sample_function(5)

        assert result == 10

        # Check that a span was created
        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == f"{__name__}.{sample_function.__qualname__}"
        assert span.status.status_code == StatusCode.OK

    @pytest.mark.asyncio
    async def test_trace_async_function_creates_span(self, in_memory_exporter):
        """Test that the @trace decorator creates a span for async functions."""
        @trace()
        async def async_sample_function(x: int) -> int:
            return x * 3

        result = await async_sample_function(4)

        assert result == 12

        # Check that a span was created
        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == f"{__name__}.{async_sample_function.__qualname__}"
        assert span.status.status_code == StatusCode.OK

    def test_trace_function_with_custom_name(self, in_memory_exporter):
        """Test that the decorator uses a custom span name when provided."""
        @trace(name="custom-operation")
        def sample_function() -> str:
            return "hello"

        result = sample_function()

        assert result == "hello"

        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "custom-operation"

    def test_trace_function_with_attributes(self, in_memory_exporter):
        """Test that the decorator adds custom attributes to the span."""
        @trace(attributes={"user_id": "123", "operation_type": "calculation"})
        def sample_function(x: int) -> int:
            return x + 1

        result = sample_function(10)

        assert result == 11

        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        attributes = dict(span.attributes or {})
        assert attributes.get("user_id") == "123"
        assert attributes.get("operation_type") == "calculation"

    def test_trace_function_captures_sync_exception(self, in_memory_exporter):
        """Test that the decorator captures exceptions in sync functions."""
        @trace()
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_function()

        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR
        assert "Test error" in span.status.description

        # Check that the exception was recorded
        events = span.events
        assert len(events) == 1
        assert events[0].name == "exception"

    @pytest.mark.asyncio
    async def test_trace_function_captures_async_exception(self, in_memory_exporter):
        """Test that the decorator captures exceptions in async functions."""
        @trace()
        async def async_failing_function():
            raise RuntimeError("Async test error")

        with pytest.raises(RuntimeError, match="Async test error"):
            await async_failing_function()

        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR
        assert "Async test error" in span.status.description

        # Check that the exception was recorded
        events = span.events
        assert len(events) == 1
        assert events[0].name == "exception"

    def test_trace_preserves_function_metadata(self, in_memory_exporter):
        """Test that the decorator preserves function metadata."""
        @trace()
        def documented_function(x: int, y: str) -> str:
            """This function has documentation."""
            return f"{x}-{y}"

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This function has documentation."

        result = documented_function(42, "test")
        assert result == "42-test"

    def test_trace_with_multiple_calls(self, in_memory_exporter):
        """Test that multiple calls create multiple spans."""
        @trace()
        def multi_call_function(value: int) -> int:
            return value + 1

        multi_call_function(1)
        multi_call_function(2)
        multi_call_function(3)

        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) == 3

        for span in spans:
            assert span.status.status_code == StatusCode.OK

    @pytest.mark.asyncio
    async def test_trace_nested_functions(self, in_memory_exporter):
        """Test that nested traced functions create nested spans."""
        @trace(name="inner-function")
        def inner_function(x: int) -> int:
            return x * 2

        @trace(name="outer-function")
        def outer_function(x: int) -> int:
            return inner_function(x) + 1

        result = outer_function(5)

        assert result == 11

        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) == 2

        # Find the outer and inner spans
        inner_span = next(s for s in spans if s.name == "inner-function")
        outer_span = next(s for s in spans if s.name == "outer-function")

        # The inner span should have the outer span as its parent
        assert inner_span.parent is not None
        assert inner_span.parent.span_id == outer_span.context.span_id

    def test_trace_alias(self, in_memory_exporter):
        """Test that 'trace' is an alias for 'trace_function'."""
        assert trace is trace_function

    def test_trace_with_method(self, in_memory_exporter):
        """Test that the decorator works with class methods."""
        class TestClass:
            @trace(name="test-method")
            def my_method(self, value: int) -> int:
                return value * 10

        obj = TestClass()
        result = obj.my_method(5)

        assert result == 50

        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "test-method"

    @pytest.mark.asyncio
    async def test_trace_with_async_method(self, in_memory_exporter):
        """Test that the decorator works with async class methods."""
        class TestClass:
            @trace(name="async-test-method")
            async def my_async_method(self, value: int) -> int:
                return value * 20

        obj = TestClass()
        result = await obj.my_async_method(3)

        assert result == 60

        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "async-test-method"

    def test_trace_with_complex_attributes(self, in_memory_exporter):
        """Test that the decorator handles various attribute types."""
        @trace(attributes={
            "string_attr": "value",
            "int_attr": 42,
            "float_attr": 3.14,
            "bool_attr": True,
        })
        def complex_attributes_function():
            return "done"

        result = complex_attributes_function()

        assert result == "done"

        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) == 1

        attributes = dict(spans[0].attributes or {})
        assert attributes["string_attr"] == "value"
        assert attributes["int_attr"] == 42
        assert attributes["float_attr"] == 3.14
        assert attributes["bool_attr"] is True
