from typing import cast
from unittest.mock import patch

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
