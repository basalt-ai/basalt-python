from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from basalt.observability import processors


class DummySpan:
    def __init__(self, is_recording=True, attributes=None) -> None:
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
    assert (
        processors.INSTRUMENTATION_SCOPE_KINDS["opentelemetry.instrumentation.openai.v1"]
        == "generation"
    )


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


def test_auto_instrumentation_processor_injects_prompt_from_contextvar():
    """Test that BasaltAutoInstrumentationProcessor reads _current_prompt_context and injects attributes."""
    from opentelemetry import context as otel_context

    from basalt.observability.context_managers import ROOT_SPAN_CONTEXT_KEY
    from basalt.prompts.models import _current_prompt_context

    # Create a mock span with Gemini instrumentation scope
    mock_span = MagicMock()
    mock_span.is_recording.return_value = True
    mock_scope = MagicMock()
    mock_scope.name = "opentelemetry.instrumentation.google_generativeai"
    mock_span.instrumentation_scope = mock_scope

    # Set up a root span context
    mock_root_span = MagicMock()
    ctx = otel_context.set_value(ROOT_SPAN_CONTEXT_KEY, mock_root_span)
    otel_token = otel_context.attach(ctx)

    # Set up prompt context via ContextVar
    prompt_ctx = {
        "slug": "test-prompt",
        "version": "v1.0.0",
        "tag": "latest",
        "provider": "gemini",
        "model": "gemini-2.5-flash-lite",
        "variables": {"var1": "value1"},
        "from_cache": False,
    }
    cv_token = _current_prompt_context.set(prompt_ctx)

    try:
        # Create processor and call on_start
        processor = processors.BasaltAutoInstrumentationProcessor()
        processor.on_start(mock_span, None)

        # Verify that prompt attributes were injected from ContextVar
        mock_span.set_attribute.assert_any_call("basalt.prompt.slug", "test-prompt")
        mock_span.set_attribute.assert_any_call("basalt.prompt.version", "v1.0.0")
        mock_span.set_attribute.assert_any_call("basalt.prompt.tag", "latest")
        mock_span.set_attribute.assert_any_call("basalt.prompt.model.provider", "gemini")
        mock_span.set_attribute.assert_any_call(
            "basalt.prompt.model.model", "gemini-2.5-flash-lite"
        )
        mock_span.set_attribute.assert_any_call("basalt.prompt.from_cache", False)

        # Check that variables were serialized as JSON
        import json

        calls = mock_span.set_attribute.call_args_list
        variables_call = [call for call in calls if call[0][0] == "basalt.prompt.variables"]
        assert len(variables_call) == 1
        assert json.loads(variables_call[0][0][1]) == {"var1": "value1"}
    finally:
        _current_prompt_context.reset(cv_token)
        otel_context.detach(otel_token)


def test_auto_instrumentation_processor_explicit_injection_overrides_contextvar():
    """Test that explicit injection via PENDING_INJECT_PROMPT_KEY takes precedence over ContextVar."""
    from opentelemetry import context as otel_context

    from basalt.observability.context_managers import ROOT_SPAN_CONTEXT_KEY
    from basalt.observability.processors import PENDING_INJECT_PROMPT_KEY
    from basalt.prompts.models import _current_prompt_context

    # Create a mock span
    mock_span = MagicMock()
    mock_span.is_recording.return_value = True
    mock_scope = MagicMock()
    mock_scope.name = "opentelemetry.instrumentation.openai"
    mock_span.instrumentation_scope = mock_scope

    # Set up a root span context
    mock_root_span = MagicMock()
    ctx = otel_context.set_value(ROOT_SPAN_CONTEXT_KEY, mock_root_span)

    # Set up ContextVar prompt context (should be overridden)
    contextvar_prompt = {
        "slug": "contextvar-prompt",
        "version": "v1.0.0",
        "provider": "openai",
        "model": "gpt-4",
        "from_cache": False,
    }
    cv_token = _current_prompt_context.set(contextvar_prompt)

    # Set up explicit injection (should take precedence)
    explicit_prompt = {
        "slug": "explicit-prompt",
        "version": "v2.0.0",
        "provider": "anthropic",
        "model": "claude-3",
    }
    ctx = otel_context.set_value(PENDING_INJECT_PROMPT_KEY, explicit_prompt, ctx)
    otel_token = otel_context.attach(ctx)

    try:
        # Create processor and call on_start
        processor = processors.BasaltAutoInstrumentationProcessor()
        processor.on_start(mock_span, None)

        # Verify that explicit injection won (not ContextVar)
        mock_span.set_attribute.assert_any_call("basalt.prompt.slug", "explicit-prompt")
        mock_span.set_attribute.assert_any_call("basalt.prompt.version", "v2.0.0")
        mock_span.set_attribute.assert_any_call("basalt.prompt.provider", "anthropic")
        mock_span.set_attribute.assert_any_call("basalt.prompt.model", "claude-3")

        # Verify ContextVar values were NOT used
        calls = mock_span.set_attribute.call_args_list
        call_dict = {call[0][0]: call[0][1] for call in calls}
        assert call_dict.get("basalt.prompt.slug") != "contextvar-prompt"
    finally:
        _current_prompt_context.reset(cv_token)
        otel_context.detach(otel_token)


def test_auto_instrumentation_processor_no_prompt_context():
    """Test that processor gracefully handles absence of prompt context."""
    from opentelemetry import context as otel_context

    from basalt.observability.context_managers import ROOT_SPAN_CONTEXT_KEY

    # Create a mock span
    mock_span = MagicMock()
    mock_span.is_recording.return_value = True
    mock_scope = MagicMock()
    mock_scope.name = "opentelemetry.instrumentation.openai"
    mock_span.instrumentation_scope = mock_scope

    # Set up a root span context but NO prompt context
    mock_root_span = MagicMock()
    ctx = otel_context.set_value(ROOT_SPAN_CONTEXT_KEY, mock_root_span)
    otel_token = otel_context.attach(ctx)

    try:
        # Create processor and call on_start (should not raise any exception)
        processor = processors.BasaltAutoInstrumentationProcessor()
        processor.on_start(mock_span, None)

        # Verify that no prompt attributes were set
        calls = mock_span.set_attribute.call_args_list
        prompt_calls = [call for call in calls if "basalt.prompt" in call[0][0]]
        assert len(prompt_calls) == 0
    finally:
        otel_context.detach(otel_token)


def test_auto_instrumentation_processor_prompt_context_with_optional_fields():
    """Test that processor handles prompt context with missing optional fields."""
    from opentelemetry import context as otel_context

    from basalt.observability.context_managers import ROOT_SPAN_CONTEXT_KEY
    from basalt.prompts.models import _current_prompt_context

    # Create a mock span
    mock_span = MagicMock()
    mock_span.is_recording.return_value = True
    mock_scope = MagicMock()
    mock_scope.name = "opentelemetry.instrumentation.openai"
    mock_span.instrumentation_scope = mock_scope

    # Set up a root span context
    mock_root_span = MagicMock()
    ctx = otel_context.set_value(ROOT_SPAN_CONTEXT_KEY, mock_root_span)
    otel_token = otel_context.attach(ctx)

    # Set up prompt context with only required fields (no version, tag, or variables)
    prompt_ctx = {
        "slug": "minimal-prompt",
        "version": None,
        "tag": None,
        "provider": "openai",
        "model": "gpt-4",
        "variables": None,
        "from_cache": True,
    }
    cv_token = _current_prompt_context.set(prompt_ctx)

    try:
        # Create processor and call on_start
        processor = processors.BasaltAutoInstrumentationProcessor()
        processor.on_start(mock_span, None)

        # Verify that required fields were set
        mock_span.set_attribute.assert_any_call("basalt.prompt.slug", "minimal-prompt")
        mock_span.set_attribute.assert_any_call("basalt.prompt.model.provider", "openai")
        mock_span.set_attribute.assert_any_call("basalt.prompt.model.model", "gpt-4")
        mock_span.set_attribute.assert_any_call("basalt.prompt.from_cache", True)

        # Verify that optional fields were NOT set
        calls = mock_span.set_attribute.call_args_list
        call_dict = {call[0][0]: call[0][1] for call in calls}
        assert "basalt.prompt.version" not in call_dict
        assert "basalt.prompt.tag" not in call_dict
        assert "basalt.prompt.variables" not in call_dict
    finally:
        _current_prompt_context.reset(cv_token)
        otel_context.detach(otel_token)
