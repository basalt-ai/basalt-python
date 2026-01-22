"""Pytest-style unit tests for ExperimentsClient.

These tests follow the same pattern as the prompts and datasets tests.
"""

from unittest.mock import patch

import pytest

from basalt._internal.http import HTTPResponse
from basalt.experiments.client import ExperimentsClient
from basalt.experiments.models import Experiment
from basalt.types.exceptions import BadRequestError, BasaltAPIError, UnauthorizedError


@pytest.fixture
def common_client():
    """Create a test client instance."""
    api_key = "test-api-key"
    client = ExperimentsClient(api_key=api_key)

    mock_experiment = Experiment(
        id="123",
        name="My Experiment",
        feature_slug="my-feature",
        created_at="2024-03-20T12:00:00Z",
    )

    return {
        "client": client,
        "mock_experiment": mock_experiment,
    }


def make_response(payload: dict | None, status: int = 200) -> HTTPResponse:
    return HTTPResponse(status_code=status, data=payload, headers={})


def test_create_sync_success(common_client):
    """Test synchronous experiment creation."""
    client: ExperimentsClient = common_client["client"]

    with patch("basalt.experiments.client.HTTPClient.fetch_sync") as mock_fetch:
        mock_fetch.return_value = make_response(
            {
                "id": "123",
                "name": "My Experiment",
                "featureSlug": "my-feature",
                "createdAt": "2024-03-20T12:00:00Z",
            }
        )

        experiment = client.create_sync(
            feature_slug="my-feature",
            name="My Experiment",
        )

        # Verify API was called
        mock_fetch.assert_called_once()
        call_kwargs = mock_fetch.call_args[1]
        assert "/monitor/experiments" in call_kwargs["url"]
        assert call_kwargs["method"] == "POST"
        assert call_kwargs["body"]["featureSlug"] == "my-feature"
        assert call_kwargs["body"]["name"] == "My Experiment"

        # Verify experiment object
        assert isinstance(experiment, Experiment)
        assert experiment.id == "123"
        assert experiment.name == "My Experiment"
        assert experiment.feature_slug == "my-feature"
        assert experiment.created_at == "2024-03-20T12:00:00Z"


def test_create_sync_error(common_client):
    """Test error handling in synchronous creation."""
    client: ExperimentsClient = common_client["client"]

    with patch("basalt.experiments.client.HTTPClient.fetch_sync") as mock_fetch:
        mock_fetch.side_effect = BadRequestError("Invalid request")

        with pytest.raises(BadRequestError):
            client.create_sync(
                feature_slug="my-feature",
                name="My Experiment",
            )


def test_create_sync_api_error_response(common_client):
    """Test handling of error in API response."""
    client: ExperimentsClient = common_client["client"]

    with patch("basalt.experiments.client.HTTPClient.fetch_sync") as mock_fetch:
        # HTTPClient now raises exceptions for error status codes
        # Simulate a 400 Bad Request error
        mock_fetch.side_effect = BadRequestError("Feature not found")

        with pytest.raises(BadRequestError) as exc_info:
            client.create_sync(
                feature_slug="nonexistent-feature",
                name="My Experiment",
            )

        assert "Feature not found" in str(exc_info.value)


def test_create_sync_empty_response(common_client):
    """Test handling of empty response."""
    client: ExperimentsClient = common_client["client"]

    with patch("basalt.experiments.client.HTTPClient.fetch_sync") as mock_fetch:
        mock_fetch.return_value = None

        with pytest.raises(BasaltAPIError):
            client.create_sync(
                feature_slug="my-feature",
                name="My Experiment",
            )


@pytest.mark.asyncio
async def test_create_async_success(common_client):
    """Test asynchronous experiment creation."""
    client: ExperimentsClient = common_client["client"]

    with patch("basalt.experiments.client.HTTPClient.fetch") as mock_fetch:
        mock_fetch.return_value = make_response(
            {
                "id": "456",
                "name": "Async Experiment",
                "featureSlug": "async-feature",
                "createdAt": "2024-03-21T10:30:00Z",
            }
        )

        experiment = await client.create(
            feature_slug="async-feature",
            name="Async Experiment",
        )

        # Verify API was called
        mock_fetch.assert_called_once()
        call_kwargs = mock_fetch.call_args[1]
        assert "/monitor/experiments" in call_kwargs["url"]
        assert call_kwargs["method"] == "POST"
        assert call_kwargs["body"]["featureSlug"] == "async-feature"
        assert call_kwargs["body"]["name"] == "Async Experiment"

        # Verify experiment object
        assert isinstance(experiment, Experiment)
        assert experiment.id == "456"
        assert experiment.name == "Async Experiment"
        assert experiment.feature_slug == "async-feature"
        assert experiment.created_at == "2024-03-21T10:30:00Z"


@pytest.mark.asyncio
async def test_create_async_error(common_client):
    """Test error handling in asynchronous creation."""
    client: ExperimentsClient = common_client["client"]

    with patch("basalt.experiments.client.HTTPClient.fetch") as mock_fetch:
        mock_fetch.side_effect = UnauthorizedError("Invalid API key")

        with pytest.raises(UnauthorizedError):
            await client.create(
                feature_slug="my-feature",
                name="My Experiment",
            )


@pytest.mark.asyncio
async def test_create_async_api_error_response(common_client):
    """Test handling of error in async API response."""
    client: ExperimentsClient = common_client["client"]

    with patch("basalt.experiments.client.HTTPClient.fetch") as mock_fetch:
        # HTTPClient now raises exceptions for error status codes
        # Simulate a 401 Unauthorized error
        mock_fetch.side_effect = UnauthorizedError("Unauthorized access")

        with pytest.raises(UnauthorizedError) as exc_info:
            await client.create(
                feature_slug="my-feature",
                name="My Experiment",
            )

        assert "Unauthorized access" in str(exc_info.value)


def test_headers_include_api_key():
    """Test that headers include the API key."""
    client = ExperimentsClient(api_key="test-key")

    headers = client._get_headers()
    assert headers["Authorization"] == "Bearer test-key"


def test_headers_include_sdk_info():
    """Test that headers include SDK information."""
    client = ExperimentsClient(api_key="test-key")

    headers = client._get_headers()
    assert "X-BASALT-SDK-VERSION" in headers
    assert "X-BASALT-SDK-TYPE" in headers
    assert headers["X-BASALT-SDK-TYPE"] == "python"


def test_headers_include_content_type():
    """Test that headers include content type."""
    client = ExperimentsClient(api_key="test-key")

    headers = client._get_headers()
    assert headers["Content-Type"] == "application/json"


@pytest.mark.parametrize(
    "feature_slug,name",
    [
        ("feature-1", "Experiment 1"),
        ("feature-2", "Experiment 2"),
        ("my-feature", "My Experiment"),
        ("test-feature", "Test Experiment"),
    ],
)
def test_create_sync_parameter_combinations(common_client, feature_slug, name):
    """Test creation with various parameter combinations."""
    client: ExperimentsClient = common_client["client"]

    with patch("basalt.experiments.client.HTTPClient.fetch_sync") as mock_fetch:
        mock_fetch.return_value = make_response(
            {
                "id": "123",
                "name": name,
                "featureSlug": feature_slug,
                "createdAt": "2024-03-20T12:00:00Z",
            }
        )

        experiment = client.create_sync(feature_slug=feature_slug, name=name)

        # Verify API was called with correct parameters
        call_kwargs = mock_fetch.call_args[1]
        assert call_kwargs["body"]["featureSlug"] == feature_slug
        assert call_kwargs["body"]["name"] == name

        # Verify experiment object
        assert experiment.feature_slug == feature_slug
        assert experiment.name == name


def test_experiment_model_from_dict_with_none():
    """Test Experiment.from_dict with None input."""
    experiment = Experiment.from_dict(None)

    assert experiment.id == ""
    assert experiment.name == ""
    assert experiment.feature_slug == ""
    assert experiment.created_at == ""


def test_experiment_model_from_dict_with_empty_dict():
    """Test Experiment.from_dict with empty dict."""
    experiment = Experiment.from_dict({})

    assert experiment.id == ""
    assert experiment.name == ""
    assert experiment.feature_slug == ""
    assert experiment.created_at == ""


def test_experiment_model_from_dict_with_partial_data():
    """Test Experiment.from_dict with partial data."""
    experiment = Experiment.from_dict(
        {
            "id": "123",
            "name": "Test",
        }
    )

    assert experiment.id == "123"
    assert experiment.name == "Test"
    assert experiment.feature_slug == ""
    assert experiment.created_at == ""


def test_experiment_model_from_dict_with_wrong_types():
    """Test Experiment.from_dict handles wrong types gracefully."""
    experiment = Experiment.from_dict(
        {
            "id": 123,  # Should be string
            "name": None,  # Should be string
            "featureSlug": ["not", "a", "string"],  # Should be string
            "createdAt": True,  # Should be string
        }
    )

    # All values should be converted to empty strings due to type checking
    assert experiment.id == ""
    assert experiment.name == ""
    assert experiment.feature_slug == ""
    assert experiment.created_at == ""
