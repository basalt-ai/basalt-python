from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from basalt.observability import processors


class DummySpan:
    def __init__(self, is_recording=True, attributes=None):
        self._is_recording = is_recording
        self.attributes = attributes if attributes is not None else {}
        self.set_attributes = {}

    def is_recording(self):
        return self._is_recording

    def set_attribute(self, key, value):
        self.set_attributes[key] = value

@pytest.fixture
def mock_semconv():
    with patch("basalt.observability.processors.semconv") as mock_semconv:
        mock_semconv.BasaltSpan.EVALUATORS = "basalt.evaluators"
        yield mock_semconv

def test_no_slugs(mock_semconv):
    span = DummySpan()
    processors._merge_evaluators(cast(processors.Span, span), [])
    assert span.set_attributes == {}

def test_span_not_recording(mock_semconv):
    span = DummySpan(is_recording=False)
    processors._merge_evaluators(cast(processors.Span, span), ["foo"])
    assert span.set_attributes == {}

def test_merge_with_no_existing(mock_semconv):
    span = DummySpan(attributes={})
    processors._merge_evaluators(cast(processors.Span, span), ["foo", "bar"])
    key = mock_semconv.BasaltSpan.EVALUATORS
    assert key in span.set_attributes
    assert span.set_attributes[key] == ["foo", "bar"]

def test_merge_with_existing(mock_semconv):
    key = mock_semconv.BasaltSpan.EVALUATORS
    span = DummySpan(attributes={key: ["foo", "baz"]})
    processors._merge_evaluators(cast(processors.Span, span), ["bar", "foo"])
    # Should merge and deduplicate: ["foo", "baz", "bar"]
    assert span.set_attributes[key] == ["foo", "baz", "bar"]

def test_merge_with_empty_and_whitespace_slugs(mock_semconv):
    key = mock_semconv.BasaltSpan.EVALUATORS
    span = DummySpan(attributes={key: ["", "  ", "foo"]})
    processors._merge_evaluators(cast(processors.Span, span), ["", "bar", " "])
    assert span.set_attributes[key] == ["foo", "bar"]

def test_merge_with_non_dict_attributes(mock_semconv):
    key = mock_semconv.BasaltSpan.EVALUATORS
    span = DummySpan()
    span.attributes = None  # attributes is not a dict
    processors._merge_evaluators(cast(processors.Span, span), ["foo"])
    assert span.set_attributes[key] == ["foo"]


def test_openai_v1_scope_recognized():
    """Test that opentelemetry.instrumentation.openai.v1 is recognized as an auto-instrumentation scope."""
    assert "opentelemetry.instrumentation.openai.v1" in processors.KNOWN_AUTO_INSTRUMENTATION_SCOPES


def test_openai_v1_scope_has_generation_kind():
    """Test that opentelemetry.instrumentation.openai.v1 is mapped to GENERATION kind."""
    assert "opentelemetry.instrumentation.openai.v1" in processors.INSTRUMENTATION_SCOPE_KINDS
    assert processors.INSTRUMENTATION_SCOPE_KINDS["opentelemetry.instrumentation.openai.v1"] == "generation"


def test_auto_instrumentation_processor_sets_in_trace_for_openai_v1():
    """Test that BasaltAutoInstrumentationProcessor sets in_trace for OpenAI v1 spans."""
    from opentelemetry import context as otel_context

    from basalt.observability import semconv
    from basalt.observability.context_managers import ROOT_SPAN_CONTEXT_KEY

    # Create a mock span with OpenAI v1 instrumentation scope
    mock_span = MagicMock()
    mock_span.is_recording.return_value = True
    mock_scope = MagicMock()
    mock_scope.name = "opentelemetry.instrumentation.openai.v1"
    mock_span.instrumentation_scope = mock_scope

    # Create a mock root span and attach it to the global context
    mock_root_span = MagicMock()
    ctx = otel_context.set_value(ROOT_SPAN_CONTEXT_KEY, mock_root_span)
    token = otel_context.attach(ctx)

    try:
        # Create processor and call on_start
        processor = processors.BasaltAutoInstrumentationProcessor()
        processor.on_start(mock_span, None)  # parent_context=None means use global context

        # Verify that basalt.in_trace was set to True
        mock_span.set_attribute.assert_any_call(semconv.BasaltSpan.IN_TRACE, True)
        # Verify that basalt.span.kind was set to "generation"
        mock_span.set_attribute.assert_any_call(semconv.BasaltSpan.KIND, "generation")
    finally:
        otel_context.detach(token)
