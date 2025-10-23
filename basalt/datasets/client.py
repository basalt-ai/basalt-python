"""
Datasets API Client.

This module provides the DatasetsClient for interacting with the Basalt Datasets API.
"""
from __future__ import annotations

from typing import Any

from .._internal.exceptions import BasaltAPIError
from .._internal.http import HTTPClient
from ..config import config
from .models import Dataset, DatasetRow


class DatasetsClient:
    """
    Client for interacting with the Basalt Datasets API.

    This client provides methods to list, retrieve, and add rows to datasets.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        http_client: HTTPClient | None = None,
    ):
        """
        Initialize the DatasetsClient.

        Args:
            api_key: The Basalt API key for authentication.
            logger: Logger instance for logging operations.
            base_url: Optional base URL for the API (defaults to config value).
        """
        self._api_key = api_key
        self._base_url = base_url or config["api_url"]
        self._http_client = http_client or HTTPClient()

    async def list(self) -> list[Dataset]:
        """
        List all datasets available in the workspace.

        Returns:
            A list of Dataset objects (without rows).

        Raises:
            BasaltAPIError: If the API request fails.
            NetworkError: If a network error occurs.
        """
        url = f"{self._base_url}/datasets"

        response = await self._http_client.fetch(
            url=url,
            method="GET",
            headers=self._get_headers(),
        )

        if response is None:
            return []

        datasets_data = response.get("datasets", [])
        return [
            Dataset.from_dict(ds)
            for ds in datasets_data
            if isinstance(ds, dict)
        ]

    def list_sync(self) -> list[Dataset]:
        """
        Synchronously list all datasets available in the workspace.

        Returns:
            A list of Dataset objects (without rows).

        Raises:
            BasaltAPIError: If the API request fails.
            NetworkError: If a network error occurs.
        """
        url = f"{self._base_url}/datasets"

        response = self._http_client.fetch_sync(
            url=url,
            method="GET",
            headers=self._get_headers(),
        )

        if response is None:
            return []

        datasets_data = response.get("datasets", [])
        return [
            Dataset.from_dict(ds)
            for ds in datasets_data
            if isinstance(ds, dict)
        ]

    async def get(self, slug: str) -> Dataset:
        """
        Get a dataset by its slug.

        Args:
            slug: The slug identifier for the dataset.

        Returns:
            Dataset object with all rows included.

        Raises:
            BasaltAPIError: If the API request fails.
            NetworkError: If a network error occurs.
        """
        url = f"{self._base_url}/datasets/{slug}"

        response = await self._http_client.fetch(
            url=url,
            method="GET",
            headers=self._get_headers(),
        )

        response = response or {}
        dataset_data = response.get("dataset", {})
        if response.get("error"):
            raise BasaltAPIError(response["error"])
        dataset = Dataset.from_dict(dataset_data)
        dataset.warning = response.get("warning")
        return dataset

    def get_sync(self, slug: str) -> Dataset:
        """
        Synchronously get a dataset by its slug.

        Args:
            slug: The slug identifier for the dataset.

        Returns:
            Dataset object with all rows included.

        Raises:
            BasaltAPIError: If the API request fails.
            NetworkError: If a network error occurs.
        """
        url = f"{self._base_url}/datasets/{slug}"

        response = self._http_client.fetch_sync(
            url=url,
            method="GET",
            headers=self._get_headers(),
        )

        response = response or {}
        dataset_data = response.get("dataset", {})
        if response.get("error"):
            raise BasaltAPIError(response["error"])
        dataset = Dataset.from_dict(dataset_data)
        dataset.warning = response.get("warning")
        return dataset

    async def add_row(
        self,
        slug: str,
        values: dict[str, str],
        name: str | None = None,
        ideal_output: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[DatasetRow, str | None]:
        """
        Create a new item (row) in a dataset.

        Args:
            slug: The slug identifier for the dataset.
            values: A dictionary of column values for the dataset item.
            name: An optional name for the dataset item.
            ideal_output: An optional ideal output for the dataset item.
            metadata: An optional metadata dictionary.

        Returns:
            A tuple containing the DatasetRow and an optional warning message.

        Raises:
            BasaltAPIError: If the API request fails.
            NetworkError: If a network error occurs.
        """
        url = f"{self._base_url}/datasets/{slug}/items"

        body: dict[str, Any] = {
            "values": values,
        }
        if name is not None:
            body["name"] = name
        if ideal_output is not None:
            body["idealOutput"] = ideal_output
        if metadata is not None:
            body["metadata"] = metadata

        response = await self._http_client.fetch(
            url=url,
            method="POST",
            body=body,
            headers=self._get_headers(),
        )

        response = response or {}
        if response.get("error"):
            raise BasaltAPIError(response["error"])

        row_data = response.get("datasetRow", {})
        warning = response.get("warning")

        return DatasetRow.from_dict(row_data), warning

    def add_row_sync(
        self,
        slug: str,
        values: dict[str, str],
        name: str | None = None,
        ideal_output: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[DatasetRow, str | None]:
        """
        Synchronously create a new item (row) in a dataset.

        Args:
            slug: The slug identifier for the dataset.
            values: A dictionary of column values for the dataset item.
            name: An optional name for the dataset item.
            ideal_output: An optional ideal output for the dataset item.
            metadata: An optional metadata dictionary.

        Returns:
            A tuple containing the DatasetRow and an optional warning message.

        Raises:
            BasaltAPIError: If the API request fails.
            NetworkError: If a network error occurs.
        """
        url = f"{self._base_url}/datasets/{slug}/items"

        body: dict[str, Any] = {
            "values": values,
        }
        if name is not None:
            body["name"] = name
        if ideal_output is not None:
            body["idealOutput"] = ideal_output
        if metadata is not None:
            body["metadata"] = metadata

        response = self._http_client.fetch_sync(
            url=url,
            method="POST",
            body=body,
            headers=self._get_headers(),
        )

        response = response or {}
        if response.get("error"):
            raise BasaltAPIError(response["error"])

        row_data = response.get("datasetRow", {})
        warning = response.get("warning")

        return DatasetRow.from_dict(row_data), warning

    def _get_headers(self) -> dict[str, str]:
        """
        Get the HTTP headers for API requests.

        Returns:
            Dictionary of HTTP headers.
        """
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-BASALT-SDK-VERSION": config["sdk_version"],
            "X-BASALT-SDK-TYPE": config["sdk_type"],
        }
