"""Integration tests for DatasetsClient with real API."""
from __future__ import annotations

import os

import pytest

from basalt._internal.exceptions import NotFoundError
from basalt.datasets.client import DatasetsClient
from basalt.datasets.models import Dataset, DatasetRow
from basalt.utils.logger import Logger


@pytest.mark.skipif(
    os.getenv("BASALT_RUN_INTEGRATION_TESTS") != "1",
    reason="Integration tests disabled. Set BASALT_RUN_INTEGRATION_TESTS=1"
)
class TestDatasetsClientIntegration:
    """Integration test suite for DatasetsClient."""

    @pytest.fixture(scope="class")
    def setup_class(self):
        """Set up integration test fixtures."""
        self.api_key = os.getenv("BASALT_API_KEY")
        if not self.api_key:
            pytest.skip("BASALT_API_KEY not set")

        self.test_dataset_slug = os.getenv("BASALT_TEST_DATASET_SLUG")
        if not self.test_dataset_slug:
            pytest.skip("BASALT_TEST_DATASET_SLUG not set")

        self.logger = Logger(log_level="all")
        self.client = DatasetsClient(api_key=self.api_key, logger=self.logger)

    def test_list_sync_real_api(self, setup_class):
        """Test synchronous dataset listing."""
        datasets = setup_class.client.list_sync()

        assert isinstance(datasets, list)
        if datasets:
            assert isinstance(datasets[0], Dataset)
            assert datasets[0].slug
            assert datasets[0].name

    def test_get_sync_real_api(self, setup_class):
        """Test synchronous dataset retrieval."""
        dataset = setup_class.client.get_sync(setup_class.test_dataset_slug)

        assert isinstance(dataset, Dataset)
        assert dataset.slug == setup_class.test_dataset_slug
        assert isinstance(dataset.rows, list)

    def test_get_sync_not_found(self, setup_class):
        """Test 404 error handling."""
        with pytest.raises(NotFoundError):
            setup_class.client.get_sync("nonexistent-dataset-12345")

    def test_add_row_sync_real_api(self, setup_class):
        """Test adding a row to a dataset."""
        # Get dataset to know columns
        dataset = setup_class.client.get_sync(setup_class.test_dataset_slug)

        if not dataset.columns:
            pytest.skip("Test dataset has no columns")

        # Create test values
        values = {col: f"test_value_{col}" for col in dataset.columns[:2]}

        row, warning = setup_class.client.add_row_sync(
            slug=setup_class.test_dataset_slug,
            values=values,
            name="integration_test_row",
            metadata={"test": "integration"},
        )

        assert isinstance(row, DatasetRow)
        assert row.values == values

    @pytest.mark.asyncio
    async def test_list_async_real_api(self, setup_class):
        """Test async dataset listing."""
        datasets = await setup_class.client.list()

        assert isinstance(datasets, list)

    @pytest.mark.asyncio
    async def test_get_async_real_api(self, setup_class):
        """Test async dataset retrieval."""
        dataset = await setup_class.client.get(setup_class.test_dataset_slug)

        assert isinstance(dataset, Dataset)

    @pytest.mark.asyncio
    async def test_add_row_async_real_api(self, setup_class):
        """Test async row addition."""
        dataset = await setup_class.client.get(setup_class.test_dataset_slug)

        if not dataset.columns:
            pytest.skip("No columns")

        values = {dataset.columns[0]: "async_test"}
        row, _ = await setup_class.client.add_row(
            slug=setup_class.test_dataset_slug,
            values=values,
        )

        assert isinstance(row, DatasetRow)
