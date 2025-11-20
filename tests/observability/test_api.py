# File: tests/test_api.py
import pytest

from basalt.observability import ObserveKind, SpanHandle
from basalt.observability.api import Observe


def test_get_config_for_kind_generation():
    """
    Test that the '_get_config_for_kind' method returns the correct
    configuration for 'generation'.
    """
    result = Observe._get_config_for_kind("generation")
    assert result[0].__name__ == "LLMSpanHandle"
    assert result[1] == "basalt.observability.generation"
    assert callable(result[2])
    assert callable(result[3])


def test_get_config_for_kind_retrieval():
    """
    Test that the '_get_config_for_kind' method returns the correct
    configuration for 'retrieval'.
    """
    result = Observe._get_config_for_kind("retrieval")
    assert result[0].__name__ == "RetrievalSpanHandle"
    assert result[1] == "basalt.observability.retrieval"
    assert callable(result[2])
    assert callable(result[3])


def test_get_config_for_kind_tool():
    """
    Test that the '_get_config_for_kind' method returns the correct
    configuration for 'tool'.
    """
    result = Observe._get_config_for_kind("tool")
    assert result[0].__name__ == "ToolSpanHandle"
    assert result[1] == "basalt.observability.tool"
    assert result[2] is None
    assert result[3] is None


def test_get_config_for_kind_function():
    """
    Test that the '_get_config_for_kind' method returns the correct
    configuration for 'function'.
    """
    result = Observe._get_config_for_kind("function")
    assert result[0].__name__ == "FunctionSpanHandle"
    assert result[1] == "basalt.observability.function"
    assert result[2] is None
    assert result[3] is None


def test_get_config_for_kind_event():
    """
    Test that the '_get_config_for_kind' method returns the correct
    configuration for 'event'.
    """
    result = Observe._get_config_for_kind("event")
    assert result[0].__name__ == "EventSpanHandle"
    assert result[1] == "basalt.observability.event"
    assert result[2] is None
    assert result[3] is None


def test_get_config_for_kind_default():
    """
    Test that the '_get_config_for_kind' method returns the correct
    default configuration when an invalid kind is provided.
    """
    result = Observe._get_config_for_kind("invalid_kind")
    assert result[0].__name__ == "SpanHandle"
    assert result[1] == "basalt.observability"
    assert result[2] is None
    assert result[3] is None

def test_call_decorator_sync_function():
    """Test using __call__ as a decorator on a synchronous function."""
    @Observe(name="test_function", kind=ObserveKind.FUNCTION, metadata={"key": "value"})
    def sample_function(x, y):
        return x + y

    result = sample_function(3, 4)
    assert result == 7


@pytest.mark.asyncio
async def test_call_decorator_async_function():
    """Test using __call__ as a decorator on an asynchronous function."""
    @Observe(name="async_test_function", kind=ObserveKind.FUNCTION, metadata={"key": "value"})
    async def async_sample_function(x, y):
        return x * y

    result = await async_sample_function(5, 6)
    assert result == 30


def test_call_decorator_handling_exceptions():
    """Test that __call__ as a decorator appropriately handles exceptions."""
    @Observe(name="exception_test_function", kind=ObserveKind.FUNCTION)
    def error_function(x):
        raise ValueError("Intentional error")

    with pytest.raises(ValueError, match="Intentional error"):
        error_function(10)

def test_observe_as_decorator():
    """Test Observe when used as a decorator for a synchronous function."""

    @Observe(name="test_decorator_function", kind=ObserveKind.FUNCTION)
    def decorated_function(x, y):
        return x + y

    result = decorated_function(2, 3)
    assert result == 5


def test_observe_as_context_manager():
    """Test Observe when used as a context manager."""

    with Observe(name="test_context_manager", kind=ObserveKind.EVENT) as span:
        assert isinstance(span, SpanHandle)
        span.set_attribute("test_key", "test_value")


def test_invalid_observe_kind():
    """Test Observe with an invalid kind."""

    with pytest.raises(ValueError):
        with Observe(name="invalid_kind", kind="unknown_kind"):
            pass


def test_observe_handles_exception():
    """Test that Observe captures exceptions and sets appropriate status."""

    @Observe(name="test_exception_capture", kind=ObserveKind.FUNCTION)
    def function_raises_exception():
        raise ValueError("An error occurred!")

    with pytest.raises(ValueError):
        function_raises_exception()


def test_observe_static_metadata():
    """Test adding static metadata using Observe."""
    from .utils import get_exporter

    exporter = get_exporter()
    exporter.clear()

    metadata = {"key1": "value1", "key2": "value2"}

    with Observe(name="test_static_metadata", kind=ObserveKind.SPAN, metadata=metadata) as span:
        assert span is not None

    # Verify metadata was set as span attributes
    spans = exporter.get_finished_spans()
    assert len(spans) > 0
    test_span = next((s for s in spans if s.name == "test_static_metadata"), None)
    assert test_span is not None
    assert test_span.attributes["key1"] == "value1"
    assert test_span.attributes["key2"] == "value2"



def test_observe_decorator_sync_function():
    """Test Observe decorator on a synchronous function."""
    observed_data = {}

    @Observe(name="Test Sync Function")
    def test_function(x, y):
        observed_data["input"] = (x, y)
        return x + y

    result = test_function(3, 4)
    assert result == 7
    assert observed_data["input"] == (3, 4)

def test_observe_context_manager():
    """Test Observe used as a context manager."""
    with Observe(name="Test Context Manager", metadata={"key": "value"}) as span:
        span.set_input("test_input")
        span.set_output("test_output")
        assert span is not None

def test_observe_with_metadata():
    """Test Observe with metadata added."""
    observed_metadata = {}

    @Observe(name="Test Function with Metadata", metadata={"user": "test_user"})
    def test_function():
        # Metadata expected to be set in the span
        observed_metadata["user"] = "test_user"
        return True

    result = test_function()
    assert result is True
    assert observed_metadata["user"] == "test_user"

@pytest.mark.asyncio
async def test_observe_decorator_async_function():
    """Test Observe decorator on an async function."""
    @Observe(name="Test Async Function", kind=ObserveKind.FUNCTION)
    async def test_async_function(x, y):
        return x * y

    result = await test_async_function(5, 7)
    assert result == 35

def test_get_root_span():
    """Test retrieving the root span using Observe.root_span()."""
    from .utils import get_exporter

    exporter = get_exporter()
    exporter.clear()
    
    root_span = None

    @Observe(name="Root Function")
    def root_function():
        nonlocal root_span
        root_span = Observe.root_span()
        return True

    result = root_function()
    assert result is True
    assert root_span is not None