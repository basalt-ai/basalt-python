"""Unit tests for the DatasetsClient using pytest."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from basalt._internal.exceptions import BadRequestError, BasaltAPIError, NotFoundError
from basalt.datasets.client import DatasetsClient
from basalt.datasets.models import Dataset, DatasetRow


class TestDatasetsClientSync:
    """Test suite for synchronous DatasetsClient methods."""

    @pytest.fixture
    def client(self):
        """Create a DatasetsClient instance for testing."""
        logger = MagicMock()
        return DatasetsClient(api_key="test-api-key", logger=logger)

    @patch("basalt.datasets.client.HTTPClient.fetch_sync")
    def test_list_sync_success(self, mock_fetch, client):
        """Test successful datasets listing."""
        mock_fetch.return_value = {"datasets": [
            {"slug": "dataset-1", "name": "Dataset 1", "columns": ["col1", "col2"]},
            {"slug": "dataset-2", "name": "Dataset 2", "columns": ["col3"]},
        ]}

        datasets = client.list_sync()

        assert len(datasets) == 2
        assert isinstance(datasets[0], Dataset)
        assert datasets[0].slug == "dataset-1"
        assert datasets[0].name == "Dataset 1"
        assert datasets[0].columns == ["col1", "col2"]
        assert datasets[0].rows == []

    @patch("basalt.datasets.client.HTTPClient.fetch_sync")
    def test_list_sync_error(self, mock_fetch, client):
        """Test list method error handling."""
        mock_fetch.side_effect = BadRequestError("Invalid request")

        with pytest.raises(BadRequestError):
            client.list_sync()

    @patch("basalt.datasets.client.HTTPClient.fetch_sync")
    def test_get_sync_success(self, mock_fetch, client):
        """Test successful dataset retrieval."""
        mock_fetch.return_value = {"dataset": {
            "slug": "test-dataset",
            "name": "Test Dataset",
            "columns": ["input", "output"],
            "rows": [
                {"values": {"input": "hello", "output": "world"}, "name": "row1"},
            ],
        }}

        dataset = client.get_sync("test-dataset")

        assert isinstance(dataset, Dataset)
        assert dataset.slug == "test-dataset"
        assert len(dataset.rows) == 1
        assert dataset.rows[0].values == {"input": "hello", "output": "world"}

    @patch("basalt.datasets.client.HTTPClient.fetch_sync")
    def test_get_sync_with_error_response(self, mock_fetch, client):
        """Test get with error in response."""
        mock_fetch.return_value = {"error": "Dataset not found"}

        with pytest.raises(BasaltAPIError):
            client.get_sync("nonexistent")

    @patch("basalt.datasets.client.HTTPClient.fetch_sync")
    def test_get_sync_api_error(self, mock_fetch, client):
        """Test get method with API error."""
        mock_fetch.side_effect = NotFoundError("Not found")

        with pytest.raises(NotFoundError):
            client.get_sync("test-dataset")

    @patch("basalt.datasets.client.HTTPClient.fetch_sync")
    def test_add_row_sync_success(self, mock_fetch, client):
        """Test successful row addition."""
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

        assert isinstance(row, DatasetRow)
        assert row.values == {"col1": "value1", "col2": "value2"}
        assert row.name == "test-row"
        assert row.ideal_output == "expected"
        assert warning is None

        # Verify API call
        call_kwargs = mock_fetch.call_args[1]
        assert "/datasets/test-dataset/items" in call_kwargs["url"]
        assert call_kwargs["body"]["values"] == {"col1": "value1", "col2": "value2"}

    @patch("basalt.datasets.client.HTTPClient.fetch_sync")
    def test_add_row_sync_with_warning(self, mock_fetch, client):
        """Test row addition with warning."""
        mock_fetch.return_value = {
            "datasetRow": {"values": {"col1": "value1"}},
            "warning": "Some warning message",
        }

        row, warning = client.add_row_sync(
            slug="test-dataset",
            values={"col1": "value1"},
        )

        assert warning == "Some warning message"

    @patch("basalt.datasets.client.HTTPClient.fetch_sync")
    def test_add_row_sync_with_error(self, mock_fetch, client):
        """Test add_row with error in response."""
        mock_fetch.return_value = {"error": "Invalid column"}

        with pytest.raises(BasaltAPIError):
            client.add_row_sync("test-dataset", {"invalid": "col"})

    def test_headers_include_api_key(self, client):
        """Test that headers include API key."""
        headers = client._get_headers()
        assert headers["Authorization"] == "Bearer test-api-key"

    def test_headers_include_sdk_info(self, client):
        """Test that headers include SDK information."""
        headers = client._get_headers()
        assert "X-BASALT-SDK-VERSION" in headers
        assert "X-BASALT-SDK-TYPE" in headers


@pytest.mark.asyncio
class TestDatasetsClientAsync:
    """Test suite for asynchronous DatasetsClient methods."""

    @pytest.fixture
    def client(self):
        """Create a DatasetsClient instance for testing."""
        logger = MagicMock()
        return DatasetsClient(api_key="test-api-key", logger=logger)

    @patch("basalt.datasets.client.HTTPClient.fetch")
    async def test_list_async_success(self, mock_fetch, client):
        """Test successful async datasets listing."""
        mock_fetch.return_value = {"datasets": [
            {"slug": "ds1", "name": "Dataset 1", "columns": ["a", "b"]},
        ]}

        datasets = await client.list()

        assert len(datasets) == 1
        assert datasets[0].slug == "ds1"

    @patch("basalt.datasets.client.HTTPClient.fetch")
    async def test_get_async_success(self, mock_fetch, client):
        """Test successful async dataset retrieval."""
        mock_fetch.return_value = {"dataset": {
            "slug": "test",
            "name": "Test",
            "columns": ["col1"],
            "rows": [],
        }}

        dataset = await client.get("test")

        assert dataset.slug == "test"
        assert isinstance(dataset, Dataset)

    @patch("basalt.datasets.client.HTTPClient.fetch")
    async def test_add_row_async_success(self, mock_fetch, client):
        """Test successful async row addition."""
        mock_fetch.return_value = {
            "datasetRow": {"values": {"col1": "val1"}},
            "warning": None,
        }

        row, warning = await client.add_row("test", {"col1": "val1"})

        assert row.values == {"col1": "val1"}
        assert warning is None

    @patch("basalt.datasets.client.HTTPClient.fetch")
    async def test_list_async_error(self, mock_fetch, client):
        """Test async list with error."""
        mock_fetch.side_effect = BadRequestError("Error")

        with pytest.raises(BadRequestError):
            await client.list()

    @patch("basalt.datasets.client.HTTPClient.fetch")
    async def test_get_async_with_error_response(self, mock_fetch, client):
        """Test async get with error in response."""
        mock_fetch.return_value = {"error": "Not found"}

        with pytest.raises(BasaltAPIError):
            await client.get("missing")

    @patch("basalt.datasets.client.HTTPClient.fetch")
    async def test_add_row_async_with_error(self, mock_fetch, client):
        """Test async add_row with error."""
        mock_fetch.return_value = {"error": "Invalid"}

        with pytest.raises(BasaltAPIError):
            await client.add_row("test", {"col": "val"})
