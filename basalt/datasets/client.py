"""
Datasets API Client.

This module provides the DatasetsClient for interacting with the Basalt Datasets API.
"""
from __future__ import annotations

from typing import Any

from .._internal.base_client import BaseServiceClient
from .._internal.http import HTTPClient
from ..config import config
from ..types.exceptions import BasaltAPIError
from .models import Dataset, DatasetRow


class DatasetsClient(BaseServiceClient):
    """
    Client for interacting with the Basalt Datasets API.

    This client provides methods to list, retrieve, and add rows to datasets.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        http_client: HTTPClient | None = None,
        log_level: str | None = None,
    ):
        """
        Initialize the DatasetsClient.

        Args:
            api_key: The Basalt API key for authentication.
            base_url: Optional base URL for the API (defaults to config value).
            log_level: Optional log level for the client logger.
        """
        self._api_key = api_key
        self._base_url = base_url or config.get("api_url")
        super().__init__(client_name="datasets", http_client=http_client, log_level=log_level)

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

        response = await self._request_async(
            "list",
            method="GET",
            url=url,
            headers=self._get_headers(),
        )

        if response is None or response.body is None:
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

        response = self._request_sync(
            "list",
            method="GET",
            url=url,
            headers=self._get_headers(),
        )

        if response is None or response.body is None:
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

        response = await self._request_async(
            "get",
            method="GET",
            url=url,
            headers=self._get_headers(),
            span_attributes={"basalt.dataset.slug": slug},
        )

        if response is None:
            raise BasaltAPIError("Empty response from dataset API")

        payload = response.json() or {}

        dataset_data = payload.get("dataset", {})
        dataset = Dataset.from_dict(dataset_data)

        # Log warning if present
        if warning := payload.get("warning"):
            self._logger.warning("Dataset API warning: %s", warning)

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

        response = self._request_sync(
            "get",
            method="GET",
            url=url,
            headers=self._get_headers(),
            span_attributes={"basalt.dataset.slug": slug},
        )

        if response is None:
            raise BasaltAPIError("Empty response from dataset API")

        payload = response.json() or {}

        dataset_data = payload.get("dataset", {})
        dataset = Dataset.from_dict(dataset_data)

        # Log warning if present
        if warning := payload.get("warning"):
            self._logger.warning("Dataset API warning: %s", warning)

        return dataset

    async def add_row(
        self,
        slug: str,
        values: dict[str, str],
        name: str | None = None,
        ideal_output: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetRow:
        """
        Create a new item (row) in a dataset.

        Args:
            slug: The slug identifier for the dataset.
            values: A dictionary of column values for the dataset item.
            name: An optional name for the dataset item.
            ideal_output: An optional ideal output for the dataset item.
            metadata: An optional metadata dictionary.

        Returns:
            The created DatasetRow.

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

        response = await self._request_async(
            "add_row",
            method="POST",
            url=url,
            body=body,
            headers=self._get_headers(),
            span_attributes={
                "basalt.dataset.slug": slug,
                "basalt.dataset.row_name": name,
            },
        )

        if response is None:
            raise BasaltAPIError("Empty response from dataset add row API")

        payload = response.json() or {}

        row_data = payload.get("datasetRow", {})

        # Log warning if present
        if warning := payload.get("warning"):
            self._logger.warning("Dataset API warning: %s", warning)

        return DatasetRow.from_dict(row_data)

    def add_row_sync(
        self,
        slug: str,
        values: dict[str, str],
        name: str | None = None,
        ideal_output: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetRow:
        """
        Synchronously create a new item (row) in a dataset.

        Args:
            slug: The slug identifier for the dataset.
            values: A dictionary of column values for the dataset item.
            name: An optional name for the dataset item.
            ideal_output: An optional ideal output for the dataset item.
            metadata: An optional metadata dictionary.

        Returns:
            The created DatasetRow.

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

        response = self._request_sync(
            "add_row",
            method="POST",
            url=url,
            body=body,
            headers=self._get_headers(),
            span_attributes={
                "basalt.dataset.slug": slug,
                "basalt.dataset.row_name": name,
            },
        )

        if response is None:
            raise BasaltAPIError("Empty response from dataset add row API")

        payload = response.json() or {}

        row_data = payload.get("datasetRow", {})

        # Log warning if present
        if warning := payload.get("warning"):
            self._logger.warning("Dataset API warning: %s", warning)

        return DatasetRow.from_dict(row_data)

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
