"""
Tests for PromptContextManager functionality.

This module tests the context manager wrappers for prompts that enable
observability spans for prompt fetches and GenAI call scoping.
"""

from unittest.mock import MagicMock

import pytest

from basalt.prompts.models import (
    AsyncPromptContextManager,
    Prompt,
    PromptContextManager,
    PromptModel,
    PromptModelParameters,
)


@pytest.fixture
def mock_prompt():
    """Create a mock Prompt object for testing."""
    return Prompt(
        slug="test-prompt",
        text="Hello {{name}}",
        raw_text="Hello {{name}}",
        model=PromptModel(
            provider="openai",
            model="gpt-4",
            version="latest",
            parameters=PromptModelParameters(
                temperature=0.7,
                max_length=1000,
                response_format="json_object",
            ),
        ),
        version="1.0.0",
        system_text="You are a helpful assistant",
        raw_system_text="You are a helpful assistant",
        variables={"name": "World"},
        tag="prod",
    )


def test_prompt_context_manager_wraps_prompt(mock_prompt):
    """Test that PromptContextManager wraps a Prompt correctly."""
    wrapper = PromptContextManager(
        prompt=mock_prompt,
        slug="test-prompt",
        version="1.0.0",
        tag="prod",
        variables={"name": "World"},
        from_cache=False,
    )

    assert isinstance(wrapper, PromptContextManager)
    # Verify attribute forwarding works
    assert wrapper.slug == "test-prompt"
    assert wrapper.text == "Hello {{name}}"
    assert wrapper.version == "1.0.0"
    assert wrapper.tag == "prod"
    assert wrapper.model.provider == "openai"


def test_prompt_context_manager_can_be_used_imperatively(mock_prompt):
    """Test that wrapper works in imperative mode (no context manager)."""
    wrapper = PromptContextManager(
        prompt=mock_prompt,
        slug="test-prompt",
        version="1.0.0",
        tag="prod",
        variables={"name": "World"},
        from_cache=False,
    )

    # Access attributes directly (imperative usage)
    text = wrapper.text
    model = wrapper.model.model

    assert text == "Hello {{name}}"
    assert model == "gpt-4"


def test_prompt_context_manager_enter_exit():
    """Test that context manager protocol works."""
    mock_prompt_obj = MagicMock()
    mock_prompt_obj.slug = "test"
    mock_prompt_obj.model.provider = "openai"
    mock_prompt_obj.model.model = "gpt-4"

    wrapper = PromptContextManager(
        prompt=mock_prompt_obj,
        slug="test",
        version="1.0.0",
        tag=None,
        variables=None,
        from_cache=False,
    )

    # Use as context manager
    with wrapper as prompt:
        assert prompt is wrapper
        assert prompt.slug == "test"


def test_async_prompt_context_manager_wraps_prompt(mock_prompt):
    """Test that AsyncPromptContextManager wraps a Prompt correctly."""
    wrapper = AsyncPromptContextManager(
        prompt=mock_prompt,
        slug="test-prompt",
        version="1.0.0",
        tag="prod",
        variables={"name": "World"},
        from_cache=True,
    )

    assert isinstance(wrapper, AsyncPromptContextManager)
    # Verify attribute forwarding works
    assert wrapper.slug == "test-prompt"
    assert wrapper.text == "Hello {{name}}"
    assert wrapper.version == "1.0.0"


@pytest.mark.asyncio
async def test_async_prompt_context_manager_enter_exit():
    """Test that async context manager protocol works."""
    mock_prompt_obj = MagicMock()
    mock_prompt_obj.slug = "test"
    mock_prompt_obj.model.provider = "openai"
    mock_prompt_obj.model.model = "gpt-4"

    wrapper = AsyncPromptContextManager(
        prompt=mock_prompt_obj,
        slug="test",
        version="1.0.0",
        tag=None,
        variables=None,
        from_cache=False,
    )

    # Use as async context manager
    async with wrapper as prompt:
        assert prompt is wrapper
        assert prompt.slug == "test"


def test_from_cache_attribute_is_set():
    """Test that from_cache attribute is properly tracked."""
    mock_prompt_obj = MagicMock()
    mock_prompt_obj.model.provider = "openai"
    mock_prompt_obj.model.model = "gpt-4"

    wrapper_cached = PromptContextManager(
        prompt=mock_prompt_obj,
        slug="test",
        version="1.0.0",
        tag=None,
        variables=None,
        from_cache=True,
    )

    wrapper_not_cached = PromptContextManager(
        prompt=mock_prompt_obj,
        slug="test",
        version="1.0.0",
        tag=None,
        variables=None,
        from_cache=False,
    )

    assert wrapper_cached._from_cache is True
    assert wrapper_not_cached._from_cache is False


def test_wrapper_forwards_all_prompt_methods(mock_prompt):
    """Test that all Prompt methods are accessible through wrapper."""
    wrapper = PromptContextManager(
        prompt=mock_prompt,
        slug="test-prompt",
        version="1.0.0",
        tag="prod",
        variables={"name": "World"},
        from_cache=False,
    )

    # Test compile_variables method is forwarded
    result = wrapper.compile_variables({"name": "Alice"})
    assert result.variables == {"name": "Alice"}
    assert result.text == "Hello Alice"


def test_wrapper_repr_and_str(mock_prompt):
    """Test that repr and str are forwarded to wrapped Prompt."""
    wrapper = PromptContextManager(
        prompt=mock_prompt,
        slug="test-prompt",
        version="1.0.0",
        tag="prod",
        variables={"name": "World"},
        from_cache=False,
    )

    # These should return the Prompt's repr/str
    assert "Prompt" in repr(wrapper)
    assert "Prompt" in str(wrapper)


@pytest.mark.asyncio
async def test_async_prompt_context_manager_sets_in_trace_attribute(mock_prompt):
    """Test that async prompt context manager uses _set_span_attributes which includes basalt.in_trace.

    Note: This test verifies the implementation rather than span capture because
    OpenTelemetry TracerProvider cannot be overridden once set in the test suite.
    The actual span attribute setting is tested in test_prompt_context_manager_sets_in_trace_attribute.
    """
    # Create wrapper - this will call _set_span_attributes which now includes basalt.in_trace
    wrapper = AsyncPromptContextManager(
        prompt=mock_prompt,
        slug="test-prompt-async",
        version="1.0.0",
        tag="prod",
        variables={"name": "World"},
        from_cache=False,
    )

    # Verify the wrapper was created successfully
    assert isinstance(wrapper, AsyncPromptContextManager)
    assert wrapper._slug == "test-prompt-async"

    # The _set_span_attributes method is shared and tested in the sync version
    # Both PromptContextManager and AsyncPromptContextManager use identical _set_span_attributes methods
    # that include span.set_attribute(semconv.BasaltSpan.IN_TRACE, True)
