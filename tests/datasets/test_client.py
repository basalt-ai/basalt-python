"""Pytest-style unit tests for DatasetsClient.

These tests were converted from unittest to pytest. They keep the same
behaviour but use pytest fixtures, parametrization and asyncio support.
"""
from unittest.mock import patch

import pytest

from basalt.datasets.client import DatasetsClient
from basalt.datasets.models import Dataset, DatasetRow
from basalt.types.exceptions import BadRequestError, BasaltAPIError, NotFoundError


@pytest.fixture
def common_client():
    api_key = "test-api-key"

    client = DatasetsClient(api_key=api_key)

    return {
        "client": client,
    }


def test_list_sync_success(common_client):
    """Test successful datasets listing."""
    client = common_client["client"]

    with patch("basalt.datasets.client.HTTPClient.fetch_sync") as mock_fetch:
        mock_fetch.return_value = {"datasets": [
            {
                "slug": "dataset-1",
                "name": "Dataset 1",
                "columns": [
                    {"name": "col1", "type": "text"},
                    {"name": "col2", "type": "text"}
                ],
            },
            {
                "slug": "dataset-2",
                "name": "Dataset 2",
                "columns": [
                    {"name": "col3", "type": "number"}
                ],
            },
        ]}

        datasets = client.list_sync()

        # Verify API was called
        mock_fetch.assert_called_once()

        # Verify datasets
        assert len(datasets) == 2
        assert isinstance(datasets[0], Dataset)
        assert datasets[0].slug == "dataset-1"
        assert datasets[0].name == "Dataset 1"
        # columns are returned as objects; check names and types
        assert [c.name for c in datasets[0].columns] == ["col1", "col2"]
        assert [c.type for c in datasets[0].columns] == ["text", "text"]
        assert datasets[0].rows == []


def test_list_sync_error(common_client):
    """Test list method error handling."""
    client = common_client["client"]

    with patch("basalt.datasets.client.HTTPClient.fetch_sync") as mock_fetch:
        mock_fetch.side_effect = BadRequestError("Invalid request")

        with pytest.raises(BadRequestError):
            client.list_sync()


def test_get_sync_success(common_client):
    """Test successful dataset retrieval."""
    client = common_client["client"]

    with patch("basalt.datasets.client.HTTPClient.fetch_sync") as mock_fetch:
        mock_fetch.return_value = {
            "warning": "Some rows contained columns that do not exist in the dataset and were omitted.",
            "dataset": {
                "slug": "test-dataset",
                "name": "Test Dataset",
                "columns": [
                    {"name": "input", "type": "text"},
                    {"name": "output", "type": "text"}
                ],
                "rows": [
                    {
                        "values": {"input": "hello", "output": "world"},
                        "idealOutput": "This is the expected output",
                        "metadata": {"source": "user"},
                    }
                ],
            },
        }

        dataset = client.get_sync("test-dataset")

        # Verify API was called
        mock_fetch.assert_called_once()
        call_kwargs = mock_fetch.call_args[1]
        assert "/datasets/test-dataset" in call_kwargs["url"]

        # Verify dataset
        assert isinstance(dataset, Dataset)
        assert dataset.slug == "test-dataset"
        # warning is set from top-level response
        assert dataset.warning == "Some rows contained columns that do not exist in the dataset and were omitted."
        assert len(dataset.rows) == 1
        assert dataset.rows[0].values == {"input": "hello", "output": "world"}
        assert dataset.rows[0].ideal_output == "This is the expected output"
        assert dataset.rows[0].metadata == {"source": "user"}


def test_get_sync_with_error_response(common_client):
    """Test get with error in response."""
    client = common_client["client"]

    with patch("basalt.datasets.client.HTTPClient.fetch_sync") as mock_fetch:
        mock_fetch.return_value = {"error": "Dataset not found"}

        with pytest.raises(BasaltAPIError):
            client.get_sync("nonexistent")


def test_get_sync_api_error(common_client):
    """Test get method with API error."""
    client = common_client["client"]

    with patch("basalt.datasets.client.HTTPClient.fetch_sync") as mock_fetch:
        mock_fetch.side_effect = NotFoundError("Not found")

        with pytest.raises(NotFoundError):
            client.get_sync("test-dataset")


def test_add_row_sync_success(common_client):
    """Test successful row addition."""
    client = common_client["client"]

    with patch("basalt.datasets.client.HTTPClient.fetch_sync") as mock_fetch:
        mock_fetch.return_value = {
            "datasetRow": {
                "values": {"col1": "value1", "col2": "value2"},
                "name": "test-row",
                "idealOutput": "expected",
                "metadata": {"key": "value"},
            },
            "warning": None,
        }

        row, warning = client.add_row_sync(
            slug="test-dataset",
            values={"col1": "value1", "col2": "value2"},
            name="test-row",
            ideal_output="expected",
            metadata={"key": "value"},
        )

        # Verify API was called
        mock_fetch.assert_called_once()
        call_kwargs = mock_fetch.call_args[1]
        assert "/datasets/test-dataset/items" in call_kwargs["url"]
        assert call_kwargs["method"] == "POST"
        assert call_kwargs["body"]["values"] == {"col1": "value1", "col2": "value2"}
        assert call_kwargs["body"]["name"] == "test-row"
        assert call_kwargs["body"]["idealOutput"] == "expected"
        assert call_kwargs["body"]["metadata"] == {"key": "value"}

        # Verify response
        assert isinstance(row, DatasetRow)
        assert row.values == {"col1": "value1", "col2": "value2"}
        assert row.name == "test-row"
        assert row.ideal_output == "expected"
        assert warning is None


def test_add_row_sync_with_warning(common_client):
    """Test row addition with warning."""
    client = common_client["client"]

    with patch("basalt.datasets.client.HTTPClient.fetch_sync") as mock_fetch:
        mock_fetch.return_value = {
            "datasetRow": {"values": {"col1": "value1"}},
            "warning": "Some warning message",
        }

        row, warning = client.add_row_sync(
            slug="test-dataset",
            values={"col1": "value1"},
        )

        assert warning == "Some warning message"


def test_add_row_sync_with_error(common_client):
    """Test add_row with error in response."""
    client = common_client["client"]

    with patch("basalt.datasets.client.HTTPClient.fetch_sync") as mock_fetch:
        mock_fetch.return_value = {"error": "Invalid column"}

        with pytest.raises(BasaltAPIError):
            client.add_row_sync("test-dataset", {"invalid": "col"})


def test_headers_include_api_key(common_client):
    """Test that headers include API key."""
    client = common_client["client"]

    headers = client._get_headers()
    assert headers["Authorization"] == "Bearer test-api-key"


def test_headers_include_sdk_info(common_client):
    """Test that headers include SDK information."""
    client = common_client["client"]

    headers = client._get_headers()
    assert "X-BASALT-SDK-VERSION" in headers
    assert "X-BASALT-SDK-TYPE" in headers
    assert headers["X-BASALT-SDK-TYPE"] == "python"


def test_headers_include_content_type(common_client):
    """Test that headers include Content-Type."""
    client = common_client["client"]

    headers = client._get_headers()
    assert headers["Content-Type"] == "application/json"


@pytest.mark.asyncio
class TestDatasetsClientAsync:
    """Test suite for asynchronous DatasetsClient methods."""

    @pytest.fixture
    def common_client(self):
        api_key = "test-api-key"

        client = DatasetsClient(api_key=api_key)

        return {
            "client": client,
        }

    async def test_list_async_success(self, common_client):
        """Test successful async datasets listing."""
        client = common_client["client"]

        with patch("basalt.datasets.client.HTTPClient.fetch") as mock_fetch:
            mock_fetch.return_value = {"datasets": [
                {
                    "slug": "ds1",
                    "name": "Dataset 1",
                    "columns": [
                        {"name": "a", "type": "text"},
                        {"name": "b", "type": "text"}
                    ],
                },
            ]}

            datasets = await client.list()

            # Verify API was called
            mock_fetch.assert_called_once()

            # Verify datasets
            assert len(datasets) == 1
            assert datasets[0].slug == "ds1"
            assert [c.name for c in datasets[0].columns] == ["a", "b"]

    async def test_get_async_success(self, common_client):
        """Test successful async dataset retrieval."""
        client = common_client["client"]

        with patch("basalt.datasets.client.HTTPClient.fetch") as mock_fetch:
            mock_fetch.return_value = {
                "warning": None,
                "dataset": {
                    "slug": "test",
                    "name": "Test",
                    "columns": [
                        {"name": "col1", "type": "text"}
                    ],
                    "rows": [],
                },
            }

            dataset = await client.get("test")

            # Verify API was called
            mock_fetch.assert_called_once()

            # Verify dataset
            assert dataset.slug == "test"
            assert isinstance(dataset, Dataset)
            assert [c.name for c in dataset.columns] == ["col1"]

    async def test_add_row_async_success(self, common_client):
        """Test successful async row addition."""
        client = common_client["client"]

        with patch("basalt.datasets.client.HTTPClient.fetch") as mock_fetch:
            mock_fetch.return_value = {
                "datasetRow": {"values": {"col1": "val1"}},
                "warning": None
            }

            row, warning = await client.add_row("test", {"col1": "val1"})

            # Verify API was called
            mock_fetch.assert_called_once()

            # Verify response
            assert row.values == {"col1": "val1"}
            assert warning is None

    async def test_list_async_error(self, common_client):
        """Test async list with error."""
        client = common_client["client"]

        with patch("basalt.datasets.client.HTTPClient.fetch") as mock_fetch:
            mock_fetch.side_effect = BadRequestError("Error")

            with pytest.raises(BadRequestError):
                await client.list()

    async def test_get_async_with_error_response(self, common_client):
        """Test async get with error in response."""
        client = common_client["client"]

        with patch("basalt.datasets.client.HTTPClient.fetch") as mock_fetch:
            mock_fetch.return_value = {"error": "Not found"}

            with pytest.raises(BasaltAPIError):
                await client.get("missing")

    async def test_add_row_async_with_error(self, common_client):
        """Test async add_row with error."""
        client = common_client["client"]

        with patch("basalt.datasets.client.HTTPClient.fetch") as mock_fetch:
            mock_fetch.return_value = {"error": "Invalid"}

            with pytest.raises(BasaltAPIError):
                await client.add_row("test", {"col": "val"})
