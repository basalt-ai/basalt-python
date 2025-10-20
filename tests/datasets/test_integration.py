"""Integration tests for DatasetsClient with real API."""
from __future__ import annotations

import os
from dataclasses import dataclass

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

    def test_list_sync_real_api(self, integration_ctx: IntegrationContext) -> None:
        """Test synchronous dataset listing."""
        datasets = integration_ctx.client.list_sync()

        assert isinstance(datasets, list)
        if datasets:
            assert isinstance(datasets[0], Dataset)
            assert datasets[0].slug
            assert datasets[0].name

    def test_get_sync_real_api(self, integration_ctx: IntegrationContext) -> None:
        """Test synchronous dataset retrieval."""
        dataset = integration_ctx.client.get_sync(integration_ctx.test_dataset_slug)

        assert isinstance(dataset, Dataset)
        assert dataset.slug == integration_ctx.test_dataset_slug
        assert isinstance(dataset.rows, list)

    def test_get_sync_not_found(self, integration_ctx: IntegrationContext) -> None:
        """Test 404 error handling."""
        with pytest.raises(NotFoundError):
            integration_ctx.client.get_sync("nonexistent-dataset-12345")

    def test_add_row_sync_real_api(self, integration_ctx: IntegrationContext) -> None:
        """Test adding a row to a dataset."""
        # Get dataset to know columns
        dataset = integration_ctx.client.get_sync(integration_ctx.test_dataset_slug)
        pytest.skip("Skip to avoid polluting real dataset")

        if not dataset.columns:
            pytest.skip("Test dataset has no columns")

        # Create test values
        values: dict[str, str] = {col: f"test_value_{col}" for col in dataset.columns[:2]}

        row, warning = integration_ctx.client.add_row_sync(
            slug=integration_ctx.test_dataset_slug,
            values=values,
            name="integration_test_row",
            metadata={"test": "integration"},
        )

        assert isinstance(row, DatasetRow)
        assert row.values == values

    @pytest.mark.asyncio
    async def test_list_async_real_api(self, integration_ctx: IntegrationContext) -> None:
        """Test async dataset listing."""
        datasets = await integration_ctx.client.list()

        assert isinstance(datasets, list)

    @pytest.mark.asyncio
    async def test_get_async_real_api(self, integration_ctx: IntegrationContext) -> None:
        """Test async dataset retrieval."""
        dataset = await integration_ctx.client.get(integration_ctx.test_dataset_slug)

        assert isinstance(dataset, Dataset)

    @pytest.mark.asyncio
    async def test_add_row_async_real_api(self, integration_ctx: IntegrationContext) -> None:
        """Test async row addition."""
        dataset = await integration_ctx.client.get(integration_ctx.test_dataset_slug)
        pytest.skip("Skip to avoid polluting real dataset")

        if not dataset.columns:
            pytest.skip("No columns")

        values: dict[str, str] = {dataset.columns[0]: "async_test"}
        row, _ = await integration_ctx.client.add_row(
            slug=integration_ctx.test_dataset_slug,
            values=values,
        )

        assert isinstance(row, DatasetRow)


# Typed container for fixture return
@dataclass
class IntegrationContext:
    api_key: str
    test_dataset_slug: str
    logger: Logger
    client: DatasetsClient


@pytest.fixture(scope="class")
def integration_ctx() -> IntegrationContext:
    """Set up integration test fixtures and return a typed context."""
    api_key = os.getenv("BASALT_API_KEY")
    if not api_key:
        pytest.skip("BASALT_API_KEY not set")

    test_dataset_slug = os.getenv("BASALT_TEST_DATASET_SLUG")
    if not test_dataset_slug:
        pytest.skip("BASALT_TEST_DATASET_SLUG not set")

    logger = Logger(log_level="all")
    client = DatasetsClient(api_key=api_key, logger=logger)

    return IntegrationContext(api_key=api_key, test_dataset_slug=test_dataset_slug, logger=logger, client=client)

