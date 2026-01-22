import asyncio
from unittest.mock import patch

import pytest

from basalt.observability.context_managers import with_evaluators
from basalt.observability.decorators import evaluate


def test_evaluate_decorator_single_slug():
    """Test the evaluate decorator with a single evaluator slug."""

    @evaluate("test-slug")
    def test_function():
        return "executed"

    result = test_function()
    assert result == "executed", "Function should execute and return 'executed'."


def test_evaluate_decorator_multiple_slugs():
    """Test the evaluate decorator with multiple evaluator slugs."""

    @evaluate(["slug1", "slug2"])
    def another_function():
        return "success"

    result = another_function()
    assert result == "success", "Function should execute and return 'success'."


def test_evaluate_decorator_no_slugs():
    """Test the evaluate decorator raises an error when no slugs are provided."""
    with pytest.raises(ValueError, match="At least one evaluator slug must be provided."):

        @evaluate([])
        def empty_slugs_function():
            pass


def test_evaluate_with_metadata_callable():
    """Test the evaluate decorator with callable metadata."""

    def metadata_resolver(param):
        return {"key": param}

    @evaluate("slug")
    def function_with_metadata(param):
        return f"Metadata resolved for {param}"

    with patch(
        "basalt.observability.decorators.with_evaluators", wraps=with_evaluators
    ) as mock_with_evaluators:
        result = function_with_metadata("test-param")

    assert result == "Metadata resolved for test-param", (
        "Function should return a correctly formatted string."
    )
    mock_with_evaluators.assert_called_once()


def test_evaluate_asynchronous_function():
    """Test that the evaluate decorator works for async functions."""

    @evaluate("async-slug")
    async def async_function():
        return "async executed"

    result = asyncio.run(async_function())
    assert result == "async executed", "Async function should execute and return 'async executed'."
