"""Basalt Dataset API Example (New Exception-Based API).

This script demonstrates how to use the new exception-based Basalt Dataset API.
It showcases both synchronous and asynchronous operations with proper error handling.

Before running this script:
1. Set your API key: export BASALT_API_KEY="your-api-key"
2. Set a test dataset slug: export BASALT_TEST_DATASET_SLUG="your-dataset-slug"
3. Install dependencies: pip install basalt-sdk

Reference: See MIGRATION_GUIDE.md for details on the new exception-based API.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

from basalt._internal.exceptions import (
    BasaltAPIError,
    NotFoundError,
    UnauthorizedError,
)
from basalt.datasets.client import DatasetsClient
from basalt.datasets.models import Dataset

# Add parent directory to path to import basalt
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

os.environ["BASALT_BUILD"] = "development"


def initialize_client() -> DatasetsClient:
    """Initialize the Basalt Datasets client with API key.

    Returns:
        DatasetsClient: Configured datasets client instance.

    Raises:
        ValueError: If BASALT_API_KEY environment variable is not set.
    """
    api_key = os.getenv("BASALT_API_KEY")
    if not api_key:
        raise ValueError("BASALT_API_KEY environment variable not set")

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    return DatasetsClient(
        api_key=api_key,
    )


def example_1_list_datasets(client: DatasetsClient) -> None:
    """Example 1: List all datasets synchronously."""
    logging.info("Listing datasets synchronously")

    try:
        dataset_list = client.list_sync()

        if dataset_list:
            for _i, _dataset in enumerate(dataset_list[:3], 1):
                logging.info(f"{_i}. {_dataset.slug} - {_dataset.name}")
                logging.info(f"   Columns: {_dataset.columns}\n")
            if len(dataset_list) > 3:
                logging.info(f"   ... and {len(dataset_list) - 3} more datasets\n")
    except UnauthorizedError:
        logging.error("Unauthorized access")
    except BasaltAPIError:
        logging.error("Basalt API error occurred")


def example_2_get_dataset(client: DatasetsClient) -> None:
    """Example 2: Get a specific dataset synchronously."""
    logging.info("Getting a specific dataset synchronously")

    dataset_slug = os.getenv("BASALT_TEST_DATASET_SLUG")
    if not dataset_slug:
        logging.error("BASALT_TEST_DATASET_SLUG environment variable not set")
        return

    try:
        dataset = client.get_sync(dataset_slug)
        logging.info(f"Dataset: {dataset.name}")
        logging.info(f"Slug: {dataset.slug}")
        logging.info(f"Columns: {dataset.columns}")
        logging.info(f"Rows: {len(dataset.rows)}\n")
    except NotFoundError:
        logging.error("Dataset not found")
    except UnauthorizedError:
        logging.error("Unauthorized access")
    except BasaltAPIError:
        logging.error("Basalt API error occurred")


def example_3_list_dataset_rows(client: DatasetsClient) -> None:
    """Example 3: List rows in a dataset."""
    logging.info("Listing dataset rows")

    dataset_slug = os.getenv("BASALT_TEST_DATASET_SLUG")
    if not dataset_slug:
        logging.error("BASALT_TEST_DATASET_SLUG environment variable not set")
        return

    try:
        dataset = client.get_sync(dataset_slug)

        if dataset.rows:
            for _i, _row in enumerate(dataset.rows[:3], 1):
                logging.info(f"{_i}. Row: {_row.name or 'Unnamed'}")
                logging.info(f"   Values: {_row.values}")
                if _row.ideal_output:
                    logging.info(f"   Ideal Output: {_row.ideal_output}")
                logging.info("")
            if len(dataset.rows) > 3:
                logging.info(f"   ... and {len(dataset.rows) - 3} more rows\n")
        else:
            logging.info("Dataset has no rows\n")
    except NotFoundError:
        logging.error("Dataset not found", exc_info=True)
    except UnauthorizedError:
        logging.error("Unauthorized access", exc_info=True)
    except BasaltAPIError:
        logging.error("Basalt API error occurred", exc_info=True)


def example_4_add_dataset_row(client: DatasetsClient) -> None:
    """Example 4: Add a row to a dataset."""
    logging.info("Adding a row to a dataset")

    dataset_slug = os.getenv("BASALT_TEST_DATASET_SLUG")
    if not dataset_slug:
        logging.error("BASALT_TEST_DATASET_SLUG environment variable not set")
        return

    try:
        # First get the dataset to understand its schema
        dataset = client.get_sync(dataset_slug)

        # Create values for each column
        values = {}
        for column in dataset.columns:
            values[column] = f"test_value_{column}"

        # Add row with optional metadata
        row, warning = client.add_row_sync(
            slug=dataset_slug,
            values=values,
            name="Example Row",
            ideal_output="expected_output",
            metadata={"source": "example_script"}
        )

        logging.info("Row added successfully")
        logging.info(f"Row values: {row.values}")
        if warning:
            logging.warning(f"Warning: {warning}\n")
        else:
            logging.info("")
    except NotFoundError:
        logging.error("Dataset not found", exc_info=True)
    except UnauthorizedError:
        logging.error("Unauthorized access", exc_info=True)
    except BasaltAPIError:
        logging.error("Basalt API error occurred", exc_info=True)


def example_5_get_dataset_metadata(client: DatasetsClient) -> None:
    """Example 5: Get dataset metadata."""
    logging.info("Getting dataset metadata")

    dataset_slug = os.getenv("BASALT_TEST_DATASET_SLUG")
    if not dataset_slug:
        logging.error("BASALT_TEST_DATASET_SLUG environment variable not set")
        return

    try:
        dataset = client.get_sync(dataset_slug)

        logging.info(f"Dataset Name: {dataset.name}")
        logging.info(f"Dataset Slug: {dataset.slug}")
        logging.info(f"Columns: {', '.join(dataset.columns)}")
        logging.info(f"Total Rows: {len(dataset.rows)}\n")
    except NotFoundError:
        logging.error("Dataset not found", exc_info=True)
    except BasaltAPIError:
        logging.error("Basalt API error occurred", exc_info=True)


async def example_6_async_list_datasets(client: DatasetsClient) -> None:
    """Example 6: List all datasets asynchronously."""
    logging.info("Listing datasets asynchronously")

    try:
        dataset_list = await client.list()

        if dataset_list:
            for _i, _dataset in enumerate(dataset_list[:3], 1):
                logging.info(f"{_i}. {_dataset.slug} - {_dataset.name}")
            if len(dataset_list) > 3:
                logging.info(f"   ... and {len(dataset_list) - 3} more datasets\n")
        else:
            logging.info("No datasets found\n")
    except UnauthorizedError:
        logging.error("Unauthorized access")
    except BasaltAPIError:
        logging.error("Basalt API error occurred", exc_info=True)


async def example_7_async_get_dataset(client: DatasetsClient) -> None:
    """Example 7: Get a dataset asynchronously."""
    logging.info("Getting a dataset asynchronously")

    dataset_slug = os.getenv("BASALT_TEST_DATASET_SLUG")
    if not dataset_slug:
        logging.error("BASALT_TEST_DATASET_SLUG environment variable not set")
        return

    try:
        dataset = await client.get(dataset_slug)
        logging.info(f"Retrieved dataset: {dataset.name}")
        logging.info(f"Rows: {len(dataset.rows)}\n")
    except NotFoundError:
        logging.error("Dataset not found", exc_info=True)
    except BasaltAPIError:
        logging.error("Basalt API error occurred", exc_info=True)


async def example_8_async_add_row(client: DatasetsClient) -> None:
    """Example 8: Add a row to a dataset asynchronously."""
    logging.info("Adding a row asynchronously")

    dataset_slug = os.getenv("BASALT_TEST_DATASET_SLUG")
    if not dataset_slug:
        logging.error("BASALT_TEST_DATASET_SLUG environment variable not set")
        return

    try:
        # First get the dataset
        dataset = await client.get(dataset_slug)

        # Create values for each column
        values = {}
        for column in dataset.columns:
            values[column] = f"async_test_{column}"

        # Add row asynchronously
        row, warning = await client.add_row(
            slug=dataset_slug,
            values=values,
            name="Async Example Row",
            metadata={"source": "async_example"}
        )

        logging.info("Row added asynchronously")
        if warning:
            logging.warning(f"Warning: {warning}\n")
        else:
            logging.info("")
    except NotFoundError:
        logging.error("Dataset not found", exc_info=True)
    except BasaltAPIError:
        logging.error("Basalt API error occurred", exc_info=True)


async def example_9_concurrent_operations(client: DatasetsClient) -> None:
    """Example 9: Execute multiple async operations concurrently.

    Demonstrates concurrent execution of multiple API calls with proper
    error handling and type-safe result processing.
    """
    logging.info("Executing concurrent operations")

    dataset_slug = os.getenv("BASALT_TEST_DATASET_SLUG")
    if not dataset_slug:
        logging.error("BASALT_TEST_DATASET_SLUG environment variable not set")
        return

    try:
        # Create multiple concurrent tasks
        tasks = [
            client.list(),
            client.get(dataset_slug),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Type-safe extraction of results
        list_result = results[0]
        get_result = results[1]

        # Process list result
        if isinstance(list_result, Exception):
            logging.error(f"List operation failed: {type(list_result).__name__}: {list_result}")
        elif isinstance(list_result, list):
            logging.info(f"Retrieved {len(list_result)} datasets")
        else:
            logging.warning(f"Unexpected list result type: {type(list_result)}")

        # Process get result
        if isinstance(get_result, Exception):
            logging.error(f"Get operation failed: {type(get_result).__name__}: {get_result}")
        elif isinstance(get_result, Dataset):
            logging.info(f"Retrieved dataset: {get_result.name} ({len(get_result.rows)} rows)")
        else:
            logging.warning(f"Unexpected get result type: {type(get_result)}")

    except BasaltAPIError as e:
        logging.error(f"Basalt API error during concurrent operations: {e}", exc_info=True)
    except Exception as e:
        logging.error(f"Unexpected error during concurrent operations: {type(e).__name__}: {e}", exc_info=True)


async def run_async_examples(client: DatasetsClient) -> None:
    """Run all async examples."""
    await example_6_async_list_datasets(client)
    await example_7_async_get_dataset(client)
    await example_8_async_add_row(client)
    await example_9_concurrent_operations(client)


def main() -> None:
    """Run all examples."""

    try:
        # Initialize client
        client = initialize_client()

        # Run synchronous examples
        example_1_list_datasets(client)
        example_2_get_dataset(client)
        example_3_list_dataset_rows(client)
        example_4_add_dataset_row(client)
        example_5_get_dataset_metadata(client)

        # Run asynchronous examples
        asyncio.run(run_async_examples(client))

    except ValueError:
        logging.error("API key not set in environment variables")
        sys.exit(1)
    except Exception:
        logging.error("An unexpected error occurred", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
