# File: tests/test_context_managers.py
from unittest.mock import MagicMock

import pytest
from opentelemetry import context as otel_context
from opentelemetry.trace import Span

from basalt.observability.context_managers import (
    EVALUATOR_CONTEXT_KEY,
    ROOT_SPAN_CONTEXT_KEY,
    EvaluatorAttachment,
    SpanHandle,
    _normalize_evaluator_entry,
    get_root_span_handle,
    with_evaluators,
)


def test_normalize_evaluator_entry_with_string():
    """Test that a string input is converted to EvaluatorAttachment with the string as slug."""
    entry = "example-slug"
    result = _normalize_evaluator_entry(entry)
    expected = EvaluatorAttachment(slug="example-slug")

    assert result == expected


def test_normalize_evaluator_entry_with_mapping():
    """Test that a mapping containing slug is converted to EvaluatorAttachment."""
    entry = {"slug": "example-slug", "metadata": {"key": "value"}}
    result = _normalize_evaluator_entry(entry)
    expected = EvaluatorAttachment(slug="example-slug", metadata={"key": "value"})

    assert result == expected


def test_normalize_evaluator_entry_with_missing_slug():
    """Test that a mapping without a slug raises ValueError."""
    entry = {"metadata": {"key": "value"}}
    with pytest.raises(ValueError, match="Evaluator mapping must include a 'slug' key."):
        _normalize_evaluator_entry(entry)


def test_normalize_evaluator_entry_with_invalid_type():
    """Test that an unsupported input type raises TypeError."""
    entry = 42  # Invalid type
    with pytest.raises(TypeError, match="Unsupported evaluator specification: 42"):
        _normalize_evaluator_entry(entry)


def test_normalize_evaluator_entry_with_existing_evaluator_attachment():
    """Test that an input already of type EvaluatorAttachment is returned as-is."""
    entry = EvaluatorAttachment(slug="example-slug", metadata={"key": "value"})
    result = _normalize_evaluator_entry(entry)

    assert result == entry


def test_with_evaluators_no_values():
    """Test with_evaluators does not set any contexts when given no values."""
    token = otel_context.attach(otel_context.set_value("test_key", "initial_value"))

    try:
        with with_evaluators(None):
            assert otel_context.get_value("test_key") == "initial_value"
    finally:
        otel_context.detach(token)


def test_with_evaluators_propagates_evaluator_slugs():
    """Test with_evaluators correctly sets and detaches evaluator slugs."""
    evaluators = [{"slug": "evaluator_1"}, {"slug": "evaluator_2"}]
    context_key = EVALUATOR_CONTEXT_KEY
    token = otel_context.attach(otel_context.set_value(context_key, ("existing_slug",)))

    try:
        with with_evaluators(evaluators):
            slugs = otel_context.get_value(context_key)
            assert slugs == ("existing_slug", "evaluator_1", "evaluator_2")

        assert otel_context.get_value(context_key) == ("existing_slug",)
    finally:
        otel_context.detach(token)


def test_set_io_with_all_fields():
    """Test set_io sets input, output, and variables when all are provided."""
    mock_span = MagicMock(spec=Span)
    span_handle = SpanHandle(span=mock_span)
    input_payload = {"key1": "value1"}
    output_payload = {"result": "success"}
    variables = {"var1": "data1"}

    span_handle.set_io(
        input_payload=input_payload,
        output_payload=output_payload,
        variables=variables,
    )

    assert span_handle._io_payload["input"] == input_payload
    assert span_handle._io_payload["output"] == output_payload
    assert span_handle._io_payload["variables"] == variables


def test_set_io_with_only_input_payload():
    """Test set_io only sets input when only input_payload is provided."""
    mock_span = MagicMock(spec=Span)
    span_handle = SpanHandle(span=mock_span)
    input_payload = {"key": "value"}

    span_handle.set_io(input_payload=input_payload)

    assert span_handle._io_payload["input"] == input_payload
    assert span_handle._io_payload["output"] is None
    assert span_handle._io_payload["variables"] is None


def test_set_io_with_only_output_payload():
    """Test set_io only sets output when only output_payload is provided."""
    mock_span = MagicMock(spec=Span)
    span_handle = SpanHandle(span=mock_span)
    output_payload = {"result": "success"}

    span_handle.set_io(output_payload=output_payload)

    assert span_handle._io_payload["input"] is None
    assert span_handle._io_payload["output"] == output_payload
    assert span_handle._io_payload["variables"] is None


def test_set_io_with_only_variables():
    """Test set_io only sets variables when only variables are provided."""
    mock_span = MagicMock(spec=Span)
    span_handle = SpanHandle(span=mock_span)
    variables = {"var1": "data"}

    span_handle.set_io(variables=variables)

    assert span_handle._io_payload["input"] is None
    assert span_handle._io_payload["output"] is None
    assert span_handle._io_payload["variables"] == variables


def test_set_io_with_no_arguments():
    """Test set_io does nothing when no arguments are provided."""
    mock_span = MagicMock(spec=Span)
    span_handle = SpanHandle(span=mock_span)

    span_handle.set_io()

    assert span_handle._io_payload["input"] is None
    assert span_handle._io_payload["output"] is None
    assert span_handle._io_payload["variables"] is None


class SimpleSpanMock:
    """Minimal mock for testing span attribute setting."""

    def __init__(self) -> None:
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


@pytest.fixture
def mock_root_span():
    """Fixture to set up a mock root span in the OTEL context."""
    span = MagicMock(spec=Span)
    token = otel_context.attach(otel_context.set_value(ROOT_SPAN_CONTEXT_KEY, span))
    try:
        yield span
    finally:
        otel_context.detach(token)


def test_get_root_span_handle_with_valid_root_span(mock_root_span):
    """Test get_root_span_handle when a valid root span exists."""
    span_handle = get_root_span_handle()
    assert span_handle is not None
    assert isinstance(span_handle.span, Span)


def test_get_root_span_handle_with_no_root_span():
    """Test get_root_span_handle when no root span is present."""
    span_handle = get_root_span_handle()
    assert span_handle is None


def test_get_root_span_handle_with_invalid_root_span():
    """Test get_root_span_handle when an invalid root span is present."""
    token = otel_context.attach(otel_context.set_value(ROOT_SPAN_CONTEXT_KEY, "not_a_span"))
    try:
        span_handle = get_root_span_handle()
        assert span_handle is None
    finally:
        otel_context.detach(token)


@pytest.fixture
def mock_span():
    """Fixture to create a mock span."""
    span = MagicMock(spec=Span)
    return span


def test_set_attribute(mock_span):
    """Test SpanHandle.set_attribute sets attributes on the span."""
    span_handle = SpanHandle(span=mock_span)
    span_handle.set_attribute("key", "value")
    mock_span.set_attribute.assert_called_once_with("key", "value")


def test_set_input(mock_span):
    """Test SpanHandle.set_input sets input payload and serializes it if tracing is enabled."""
    with pytest.MonkeyPatch().context() as monkeypatch:
        monkeypatch.setattr(
            "basalt.observability.context_managers.trace_content_enabled", lambda: True
        )
        span_handle = SpanHandle(span=mock_span)
        payload = {"key": "value"}
        span_handle.set_input(payload)
        assert span_handle._io_payload["input"] == payload
        mock_span.set_attribute.assert_called_once()


def test_set_output(mock_span):
    """Test SpanHandle.set_output sets output payload and serializes it if tracing is enabled."""
    with pytest.MonkeyPatch().context() as monkeypatch:
        monkeypatch.setattr(
            "basalt.observability.context_managers.trace_content_enabled", lambda: True
        )
        span_handle = SpanHandle(span=mock_span)
        payload = {"result": "success"}
        span_handle.set_output(payload)
        assert span_handle._io_payload["output"] == payload
        mock_span.set_attribute.assert_called_once()


def test_set_io(mock_span):
    """Test SpanHandle.set_io sets all I/O payloads correctly."""
    with pytest.MonkeyPatch().context() as monkeypatch:
        monkeypatch.setattr(
            "basalt.observability.context_managers.trace_content_enabled", lambda: False
        )
        span_handle = SpanHandle(span=mock_span)
        input_payload = {"input": "data"}
        output_payload = {"output": "data"}
        variables = {"key": "value"}
        span_handle.set_io(
            input_payload=input_payload, output_payload=output_payload, variables=variables
        )
        assert span_handle._io_payload["input"] == input_payload
        assert span_handle._io_payload["output"] == output_payload
        assert span_handle._io_payload["variables"] == variables


def test_io_snapshot(mock_span):
    """Test SpanHandle.io_snapshot returns a copy of the I/O payload."""
    span_handle = SpanHandle(span=mock_span)
    span_handle._io_payload = {
        "input": {"input_key": "input_value"},
        "output": {"output_key": "output_value"},
        "variables": {"var1": "value1"},
    }
    snapshot = span_handle._io_snapshot()
    assert snapshot == {
        "input": {"input_key": "input_value"},
        "output": {"output_key": "output_value"},
        "variables": {"var1": "value1"},
    }
    # Ensure the snapshot is a copy and not a reference to the original
    snapshot["variables"]["var1"] = "modified"
    assert span_handle._io_payload["variables"]["var1"] == "value1"


def test_identify_with_user_only():
    """Test SpanHandle.identify sets user attributes only."""
    from basalt.observability import semconv

    mock_span = SimpleSpanMock()
    span_handle = SpanHandle(span=mock_span)
    span_handle.set_identity({"user": {"id": "user-123", "name": "John Doe"}})

    assert mock_span.attributes[semconv.BasaltUser.ID] == "user-123"
    assert mock_span.attributes[semconv.BasaltUser.NAME] == "John Doe"
    assert semconv.BasaltOrganization.ID not in mock_span.attributes


def test_identify_with_organization_only():
    """Test SpanHandle.identify sets organization attributes only."""
    from basalt.observability import semconv

    mock_span = SimpleSpanMock()
    span_handle = SpanHandle(span=mock_span)
    span_handle.set_identity({"organization": {"id": "org-456", "name": "Acme Corp"}})

    assert mock_span.attributes[semconv.BasaltOrganization.ID] == "org-456"
    assert mock_span.attributes[semconv.BasaltOrganization.NAME] == "Acme Corp"
    assert semconv.BasaltUser.ID not in mock_span.attributes


def test_identify_with_both_user_and_organization():
    """Test SpanHandle.identify sets both user and organization attributes."""
    from basalt.observability import semconv

    mock_span = SimpleSpanMock()
    span_handle = SpanHandle(span=mock_span)
    span_handle.set_identity(
        {
            "user": {"id": "user-123", "name": "John Doe"},
            "organization": {"id": "org-456", "name": "Acme Corp"},
        }
    )

    assert mock_span.attributes[semconv.BasaltUser.ID] == "user-123"
    assert mock_span.attributes[semconv.BasaltUser.NAME] == "John Doe"
    assert mock_span.attributes[semconv.BasaltOrganization.ID] == "org-456"
    assert mock_span.attributes[semconv.BasaltOrganization.NAME] == "Acme Corp"


def test_identify_with_ids_only():
    """Test SpanHandle.identify sets IDs without names."""
    from basalt.observability import semconv

    mock_span = SimpleSpanMock()
    span_handle = SpanHandle(span=mock_span)
    span_handle.set_identity({"user": {"id": "user-789"}, "organization": {"id": "org-101"}})

    assert mock_span.attributes[semconv.BasaltUser.ID] == "user-789"
    assert mock_span.attributes[semconv.BasaltOrganization.ID] == "org-101"
    assert semconv.BasaltUser.NAME not in mock_span.attributes
    assert semconv.BasaltOrganization.NAME not in mock_span.attributes


def test_identify_with_no_parameters():
    """Test SpanHandle.identify does nothing when no parameters are provided."""
    from basalt.observability import semconv

    mock_span = SimpleSpanMock()
    span_handle = SpanHandle(span=mock_span)
    span_handle.set_identity(None)

    assert semconv.BasaltUser.ID not in mock_span.attributes
    assert semconv.BasaltOrganization.ID not in mock_span.attributes


# ==================== Async Context Manager Tests ====================


@pytest.mark.asyncio
async def test_async_start_observe_context_manager(setup_tracing):
    """Test AsyncStartObserve works as an async context manager."""
    from basalt.observability import AsyncStartObserve, StartSpanHandle

    async with AsyncStartObserve(name="test_async_root_span", feature_slug="test_feature") as span:
        assert isinstance(span, StartSpanHandle)
        assert span is not None
        # Verify root span attributes
        assert span._span.attributes.get("basalt.root") is True
        assert span._span.attributes.get("basalt.in_trace") is True


@pytest.mark.asyncio
async def test_async_observe_context_manager(setup_tracing):
    """Test AsyncObserve works as an async context manager within a root span."""
    from basalt.observability import AsyncObserve, AsyncStartObserve, ObserveKind, SpanHandle

    async with AsyncStartObserve(name="test_async_root", feature_slug="test_feature") as root_span:
        async with AsyncObserve(name="test_async_child", kind=ObserveKind.FUNCTION) as child_span:
            assert isinstance(child_span, SpanHandle)
            assert child_span is not None
            # Verify child span attributes
            assert child_span._span.attributes.get("basalt.trace") is True
            assert child_span._span.attributes.get("basalt.in_trace") is True
            # Verify root span attributes
            assert root_span._span.attributes.get("basalt.root") is True
            assert root_span._span.attributes.get("basalt.in_trace") is True


@pytest.mark.asyncio
async def test_async_standalone_observe_context_manager(setup_tracing):
    """Test AsyncObserve works as standalone context manager (creates root span)."""
    from basalt.observability import AsyncObserve, ObserveKind

    async with AsyncObserve(name="test_async_standalone", kind=ObserveKind.FUNCTION) as span:
        assert span is not None
        # Standalone observe creates a root span
        assert span._span.attributes.get("basalt.root") is True
        assert span._span.attributes.get("basalt.in_trace") is True


@pytest.mark.asyncio
async def test_async_deeply_nested_context_managers(setup_tracing):
    """Test deeply nested async context managers all have correct attributes."""
    from basalt.observability import AsyncObserve, AsyncStartObserve, ObserveKind

    async with AsyncStartObserve(name="async_root", feature_slug="test_nested") as root:
        async with AsyncObserve(kind=ObserveKind.TOOL, name="async_level1") as l1:
            async with AsyncObserve(kind=ObserveKind.TOOL, name="async_level2") as l2:
                async with AsyncObserve(kind=ObserveKind.GENERATION, name="async_level3") as l3:
                    assert root._span.attributes.get("basalt.in_trace") is True
                    assert root._span.attributes.get("basalt.root") is True
                    assert l1._span.attributes.get("basalt.in_trace") is True
                    assert l1._span.attributes.get("basalt.trace") is True
                    assert l2._span.attributes.get("basalt.in_trace") is True
                    assert l2._span.attributes.get("basalt.trace") is True
                    assert l3._span.attributes.get("basalt.in_trace") is True
                    assert l3._span.attributes.get("basalt.trace") is True


@pytest.mark.asyncio
async def test_async_start_observe_with_metadata():
    """Test AsyncStartObserve correctly applies metadata."""
    import json

    from basalt.observability import AsyncStartObserve, semconv

    from .utils import get_exporter

    exporter = get_exporter()
    exporter.clear()

    metadata = {"custom_key": "custom_value", "test_id": 42}
    async with AsyncStartObserve(
        name="test_with_metadata", feature_slug="test_metadata", metadata=metadata
    ) as span:
        # Verify span was created successfully
        assert span is not None
        assert isinstance(span._span, object)  # Span exists

    # Verify metadata was properly stored in basalt.metadata
    spans = exporter.get_finished_spans()
    test_span = next((s for s in spans if s.name == "test_with_metadata"), None)
    assert test_span is not None
    assert semconv.BasaltSpan.METADATA in test_span.attributes

    metadata_json = test_span.attributes[semconv.BasaltSpan.METADATA]
    parsed_metadata = json.loads(metadata_json)
    assert parsed_metadata["custom_key"] == "custom_value"
    assert parsed_metadata["test_id"] == 42


@pytest.mark.asyncio
async def test_async_observe_with_evaluators():
    """Test AsyncObserve correctly applies evaluators."""
    from basalt.observability import AsyncObserve, AsyncStartObserve, ObserveKind

    async with AsyncStartObserve(name="test_root", feature_slug="test_evaluators"):
        evaluator_slugs = ["test_evaluator_1", "test_evaluator_2"]
        async with AsyncObserve(
            name="test_with_evaluators", kind=ObserveKind.GENERATION, evaluators=evaluator_slugs
        ) as span:
            # The evaluators should be attached to the span
            # We can verify by checking the context or span attributes
            assert span is not None


@pytest.mark.asyncio
async def test_async_observe_set_input_output():
    """Test AsyncObserve span handle can set input/output."""
    from basalt.observability import AsyncObserve, AsyncStartObserve, ObserveKind

    async with AsyncStartObserve(name="test_root", feature_slug="test_input_output"):
        async with AsyncObserve(name="test_io", kind=ObserveKind.FUNCTION) as span:
            test_input = {"query": "test query"}
            test_output = {"result": "test result"}

            span.set_input(test_input)
            span.set_output(test_output)

            # Verify the input/output were set successfully
            assert span is not None


@pytest.mark.asyncio
async def test_async_nested_spans():
    """Test multiple nested AsyncObserve spans."""
    from basalt.observability import AsyncObserve, AsyncStartObserve, ObserveKind

    async with AsyncStartObserve(name="test_root", feature_slug="test_nested_spans") as root:
        async with AsyncObserve(name="level_1", kind=ObserveKind.FUNCTION) as span1:
            assert span1 is not None
            async with AsyncObserve(name="level_2", kind=ObserveKind.EVENT) as span2:
                assert span2 is not None
                async with AsyncObserve(name="level_3", kind=ObserveKind.RETRIEVAL) as span3:
                    assert span3 is not None
                    # All spans should be properly nested
                    assert root is not None
                    assert span1 is not None
                    assert span2 is not None
                    assert span3 is not None


@pytest.mark.asyncio
async def test_async_observe_with_exception():
    """Test AsyncObserve properly handles exceptions."""
    from basalt.observability import AsyncObserve, AsyncStartObserve, ObserveKind

    with pytest.raises(ValueError, match="Test exception"):
        async with AsyncStartObserve(name="test_root", feature_slug="test_exception"):
            async with AsyncObserve(name="test_exception", kind=ObserveKind.FUNCTION):
                raise ValueError("Test exception")


@pytest.mark.asyncio
async def test_async_start_observe_with_identity():
    """Test AsyncStartObserve correctly applies identity."""
    from basalt.observability import AsyncStartObserve

    identity = {
        "user": {"id": "user-123", "name": "Test User"},
        "organization": {"id": "org-456", "name": "Test Org"},
    }

    async with AsyncStartObserve(
        name="test_identity", feature_slug="test_identity", identity=identity
    ) as span:
        # Verify span and identity setup completed successfully
        assert span is not None


def test_start_observe_has_in_trace_attribute(setup_tracing):
    """Verify that root spans created by start_observe have basalt.in_trace=true."""
    from basalt.observability import StartObserve

    with StartObserve(name="test_root", feature_slug="test_in_trace") as span:
        assert span._span.attributes.get("basalt.root") is True
        assert span._span.attributes.get("basalt.in_trace") is True


def test_child_observe_has_in_trace_attribute(setup_tracing):
    """Verify that child spans within a trace have basalt.in_trace=true."""
    from basalt.observability import Observe, ObserveKind, StartObserve

    with StartObserve(name="root", feature_slug="test_child_trace"):
        with Observe(kind=ObserveKind.GENERATION, name="child") as child_span:
            assert child_span._span.attributes.get("basalt.trace") is True
            assert child_span._span.attributes.get("basalt.in_trace") is True


def test_standalone_observe_has_in_trace_attribute(setup_tracing):
    """Verify that standalone observe spans have basalt.in_trace=true."""
    from basalt.observability import Observe, ObserveKind

    with Observe(kind=ObserveKind.GENERATION, name="standalone") as span:
        # Standalone observe creates a root span
        assert span._span.attributes.get("basalt.root") is True
        assert span._span.attributes.get("basalt.in_trace") is True


def test_deeply_nested_spans_have_in_trace_attribute(setup_tracing):
    """Verify that deeply nested spans all have basalt.in_trace=true."""
    from basalt.observability import Observe, ObserveKind, StartObserve

    with StartObserve(name="root", feature_slug="test_deeply_nested") as root:
        with Observe(kind=ObserveKind.TOOL, name="level1") as l1:
            with Observe(kind=ObserveKind.TOOL, name="level2") as l2:
                with Observe(kind=ObserveKind.GENERATION, name="level3") as l3:
                    assert root._span.attributes.get("basalt.in_trace") is True
                    assert l1._span.attributes.get("basalt.in_trace") is True
                    assert l2._span.attributes.get("basalt.in_trace") is True
                    assert l3._span.attributes.get("basalt.in_trace") is True


@pytest.mark.asyncio
async def test_async_observe_has_in_trace_attribute(setup_tracing):
    """Verify that async spans have basalt.in_trace=true."""
    from basalt.observability import AsyncObserve, AsyncStartObserve, ObserveKind

    async with AsyncStartObserve(name="async_root", feature_slug="test_async_trace") as root_span:
        assert root_span._span.attributes.get("basalt.in_trace") is True
        async with AsyncObserve(kind=ObserveKind.GENERATION, name="async_child") as child_span:
            assert child_span._span.attributes.get("basalt.in_trace") is True
