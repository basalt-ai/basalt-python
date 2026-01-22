from unittest.mock import MagicMock, call, patch

from basalt.observability import evaluators


class DummyAttachment:
    def __init__(self, slug) -> None:
        self.slug = slug


@patch("basalt.observability.evaluators.normalize_evaluator_specs")
def test_flatten_evaluator_specs(mock_normalize):
    mock_normalize.return_value = [
        DummyAttachment("slug1"),
        DummyAttachment("slug2"),
        None,
        DummyAttachment(123),  # Not a string, should be skipped
    ]
    result = evaluators._flatten_evaluator_specs("a", "b")
    assert result == ["slug1", "slug2"]


@patch("basalt.observability.evaluators.with_evaluators")
@patch("basalt.observability.evaluators._flatten_evaluator_specs")
def test_attach_evaluator_with_explicit_span(mock_flatten, mock_with_evaluators):
    mock_flatten.return_value = ["slug1", "slug2"]
    mock_cm = MagicMock()
    mock_with_evaluators.return_value.__enter__.return_value = mock_cm
    span = MagicMock()
    with evaluators.attach_evaluator("slug1", "slug2", span=span):
        pass
    span.add_evaluator.assert_has_calls([call("slug1"), call("slug2")])


@patch("basalt.observability.evaluators.trace.get_current_span")
@patch("basalt.observability.evaluators.SpanHandle")
@patch("basalt.observability.evaluators.with_evaluators")
@patch("basalt.observability.evaluators._flatten_evaluator_specs")
def test_attach_evaluator_with_current_span(
    mock_flatten, mock_with_evaluators, mock_spanhandle, mock_get_current_span
):
    mock_flatten.return_value = ["slugA"]
    mock_span = MagicMock()
    mock_get_current_span.return_value = mock_span
    mock_span.get_span_context.return_value.is_valid = True
    span_handle = MagicMock()
    mock_spanhandle.return_value = span_handle
    with evaluators.attach_evaluator("slugA"):
        pass
    span_handle.add_evaluator.assert_called_once_with("slugA")


@patch("basalt.observability.evaluators._flatten_evaluator_specs")
def test_attach_evaluators_to_span(mock_flatten):
    mock_flatten.return_value = ["slug1", "slug2"]
    span_handle = MagicMock()
    evaluators.attach_evaluators_to_span(span_handle, "slug1", "slug2")
    span_handle.add_evaluator.assert_has_calls([call("slug1"), call("slug2")])


@patch("basalt.observability.evaluators.trace.get_current_span")
@patch("basalt.observability.evaluators.SpanHandle")
@patch("basalt.observability.evaluators._flatten_evaluator_specs")
def test_attach_evaluators_to_current_span_valid(
    mock_flatten, mock_spanhandle, mock_get_current_span
):
    mock_flatten.return_value = ["slugX"]
    mock_span = MagicMock()
    mock_get_current_span.return_value = mock_span
    mock_span.get_span_context.return_value.is_valid = True
    span_handle = MagicMock()
    mock_spanhandle.return_value = span_handle
    evaluators.attach_evaluators_to_current_span("slugX")
    span_handle.add_evaluator.assert_called_once_with("slugX")


@patch("basalt.observability.evaluators.trace.get_current_span")
@patch("basalt.observability.evaluators._flatten_evaluator_specs")
def test_attach_evaluators_to_current_span_invalid(mock_flatten, mock_get_current_span):
    mock_flatten.return_value = ["slugY"]
    mock_span = MagicMock()
    mock_get_current_span.return_value = mock_span
    mock_span.get_span_context.return_value.is_valid = False
    # Should not raise or call anything
    evaluators.attach_evaluators_to_current_span("slugY")
    # No exception means pass
