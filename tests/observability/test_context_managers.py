# File: tests/test_context_managers.py
from unittest.mock import MagicMock, create_autospec

import pytest
from opentelemetry import context as otel_context
from opentelemetry.trace import Span, SpanContext
from opentelemetry.trace.status import StatusCode

from basalt.observability.context_managers import (
    EVALUATOR_CONTEXT_KEY,
    ROOT_SPAN_CONTEXT_KEY,
    EvaluatorAttachment,
    SpanHandle,
    _normalize_evaluator_entry,
    get_root_span_handle,
    with_evaluators,
)
from basalt.observability.semconv import BasaltExperiment


class _MockSpanContext(SpanContext):
    """Simple SpanContext stand-in that always reports as valid."""

    def __init__(
        self,
        trace_id=0x1234567890ABCDEF1234567890ABCDEF,
        span_id=0x1234567890ABCDEF,
        is_remote=False,
        trace_flags=0,
        trace_state=None,
    ):
        # Provide dummy values for required SpanContext fields
        super().__init__(
            trace_id=trace_id,      # trace_id
            span_id=span_id,        # span_id
            is_remote=is_remote,    # is_remote
            trace_flags=trace_flags,  # trace_flags
            trace_state=trace_state   # trace_state
        )

    @property
    def is_valid(self) -> bool:
        return True


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


class MockSpan(Span):
    """A mock implementation of Span for testing purposes."""

    def __init__(self):
        self.attributes = {}
        self.name = "mock-span"
        self.status = None

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def set_attributes(self, attributes):
        if isinstance(attributes, dict):
            for key, value in attributes.items():
                self.set_attribute(key, value)

    def add_event(self, name, attributes=None, timestamp=None):
        return None

    def set_status(self, status, description=None):
        self.status = status
        self.status_description = description

    def record_exception(
        self,
        exception,
        attributes=None,
        timestamp=None,
        escaped=False,
    ):
        return None

    def update_name(self, name):
        self.name = name

    def end(self, end_time=None):
        return None

    def get_span_context(self):
        return _MockSpanContext()

    def is_recording(self):
        return True


@pytest.fixture
def mock_root_span():
    """Fixture to set up a mock root span in the OTEL context."""
    span = MockSpan()
    token = otel_context.attach(otel_context.set_value(ROOT_SPAN_CONTEXT_KEY, span))
    try:
        yield span
    finally:
        otel_context.detach(token)


def test_get_root_span_handle_with_valid_root_span(mock_root_span):
    """Test get_root_span_handle when a valid root span exists."""
    span_handle = get_root_span_handle()
    assert span_handle is not None
    assert isinstance(span_handle.span, MockSpan)


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


def test_set_experiment_on_root_span():
    """Test setting experiment attributes on a root span."""
    span = create_autospec(Span, instance=True)
    span.parent = None  # No parent, making this a root span
    span_handle = SpanHandle(span)

    span_handle.set_experiment("exp-123", name="Experiment Test", feature_slug="feature-test")

    span.set_attribute.assert_any_call(BasaltExperiment.ID, "exp-123")
    span.set_attribute.assert_any_call(BasaltExperiment.NAME, "Experiment Test")
    span.set_attribute.assert_any_call(BasaltExperiment.FEATURE_SLUG, "feature-test")

def test_set_experiment_on_child_span_logs_warning(caplog):
    """Test that setting an experiment on a child span logs a warning and does not set attributes."""
    span = create_autospec(Span, instance=True)
    span.parent = MagicMock()  # Mock parent span to simulate a child span
    span.parent.is_valid = True  # Simulate valid parent

    span_handle = SpanHandle(span)

    with caplog.at_level("WARNING"):
        span_handle.set_experiment("exp-456", name="Child Experiment Test", feature_slug="child-feature")

    assert "Experiments can only be attached to root spans." in caplog.text
    span.set_attribute.assert_not_called()

def test_set_experiment_with_partial_attributes():
    """Test setting experiment attributes with only experiment ID."""
    span = create_autospec(Span, instance=True)
    span.parent = None  # No parent, making this a root span
    span_handle = SpanHandle(span)

    span_handle.set_experiment("exp-partial")

    span.set_attribute.assert_any_call(BasaltExperiment.ID, "exp-partial")
    calls = span.set_attribute.call_args_list
    assert not any(args and args[0] == BasaltExperiment.NAME for args, _ in calls)
    assert not any(args and args[0] == BasaltExperiment.FEATURE_SLUG for args, _ in calls)

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

def test_add_event(mock_span):
    """Test SpanHandle.add_event adds events to the span."""
    span_handle = SpanHandle(span=mock_span)
    span_handle.add_event("event_name", {"attr1": "value1"})
    mock_span.add_event.assert_called_once_with("event_name", attributes={"attr1": "value1"})

def test_set_status(mock_span):
    """Test SpanHandle.set_status sets the status on the span."""
    span_handle = SpanHandle(span=mock_span)
    span_handle.set_status(StatusCode.ERROR, "Something went wrong")
    mock_span.set_status.assert_called_once()
    status_arg = mock_span.set_status.call_args.args[0]
    assert status_arg.status_code == StatusCode.ERROR
    assert status_arg.description == "Something went wrong"

def test_record_exception(mock_span):
    """Test SpanHandle.record_exception records an exception on the span."""
    span_handle = SpanHandle(span=mock_span)
    exception = ValueError("Test exception")
    span_handle.record_exception(exception)
    mock_span.record_exception.assert_called_once_with(exception)

def test_set_input(mock_span):
    """Test SpanHandle.set_input sets input payload and serializes it if tracing is enabled."""
    with pytest.MonkeyPatch().context() as monkeypatch:
        monkeypatch.setattr(
            "basalt.observability.context_managers.trace_content_enabled",
            lambda: True
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
            "basalt.observability.context_managers.trace_content_enabled",
            lambda: True
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
            "basalt.observability.context_managers.trace_content_enabled",
            lambda: False
        )
        span_handle = SpanHandle(span=mock_span)
        input_payload = {"input": "data"}
        output_payload = {"output": "data"}
        variables = {"key": "value"}
        span_handle.set_io(
            input_payload=input_payload,
            output_payload=output_payload,
            variables=variables
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
    snapshot = span_handle.io_snapshot()
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

    mock_span = MockSpan()
    span_handle = SpanHandle(span=mock_span)
    span_handle.set_identity(user_id="user-123", user_name="John Doe")

    assert mock_span.attributes[semconv.BasaltUser.ID] == "user-123"
    assert mock_span.attributes[semconv.BasaltUser.NAME] == "John Doe"
    assert semconv.BasaltOrganization.ID not in mock_span.attributes


def test_identify_with_organization_only():
    """Test SpanHandle.identify sets organization attributes only."""
    from basalt.observability import semconv

    mock_span = MockSpan()
    span_handle = SpanHandle(span=mock_span)
    span_handle.set_identity(organization_id="org-456", organization_name="Acme Corp")

    assert mock_span.attributes[semconv.BasaltOrganization.ID] == "org-456"
    assert mock_span.attributes[semconv.BasaltOrganization.NAME] == "Acme Corp"
    assert semconv.BasaltUser.ID not in mock_span.attributes


def test_identify_with_both_user_and_organization():
    """Test SpanHandle.identify sets both user and organization attributes."""
    from basalt.observability import semconv

    mock_span = MockSpan()
    span_handle = SpanHandle(span=mock_span)
    span_handle.set_identity(
        user_id="user-123",
        user_name="John Doe",
        organization_id="org-456",
        organization_name="Acme Corp"
    )

    assert mock_span.attributes[semconv.BasaltUser.ID] == "user-123"
    assert mock_span.attributes[semconv.BasaltUser.NAME] == "John Doe"
    assert mock_span.attributes[semconv.BasaltOrganization.ID] == "org-456"
    assert mock_span.attributes[semconv.BasaltOrganization.NAME] == "Acme Corp"


def test_identify_with_ids_only():
    """Test SpanHandle.identify sets IDs without names."""
    from basalt.observability import semconv

    mock_span = MockSpan()
    span_handle = SpanHandle(span=mock_span)
    span_handle.set_identity(user_id="user-789", organization_id="org-101")

    assert mock_span.attributes[semconv.BasaltUser.ID] == "user-789"
    assert mock_span.attributes[semconv.BasaltOrganization.ID] == "org-101"
    assert semconv.BasaltUser.NAME not in mock_span.attributes
    assert semconv.BasaltOrganization.NAME not in mock_span.attributes


def test_identify_with_no_parameters():
    """Test SpanHandle.identify does nothing when no parameters are provided."""
    from basalt.observability import semconv

    mock_span = MockSpan()
    span_handle = SpanHandle(span=mock_span)
    span_handle.set_identity()

    assert semconv.BasaltUser.ID not in mock_span.attributes
    assert semconv.BasaltOrganization.ID not in mock_span.attributes

