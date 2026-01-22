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
from .file_upload import FileAttachment, FileUploadHandler
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
    ) -> None:
        """
        Initialize the DatasetsClient.

        Args:
            api_key: The Basalt API key for authentication.
            base_url: Optional base URL for the API (defaults to config value).
            log_level: Optional log level for the client logger.
        """
        self._api_key = api_key
        self._base_url = base_url or config["api_url"]
        super().__init__(client_name="datasets", http_client=http_client, log_level=log_level)

        # Initialize file upload handler
        self._file_upload_handler = FileUploadHandler(
            http_client=self._http_client,
            base_url=self._base_url,
            api_key=self._api_key,
        )

    @staticmethod
    def _ensure_response(response: object) -> object:
        if response is None:
            raise BasaltAPIError("Empty response from dataset API")
        return response

    def _dataset_from_response(self, response: object) -> Dataset:
        # The response is expected to be an HTTP response object with a .json() method.
        # Add a runtime check and type ignore for static analysis tools.
        response = self._ensure_response(response)
        if not hasattr(response, "json") or not callable(getattr(response, "json", None)):
            raise BasaltAPIError("Response object does not have a callable .json() method")
        payload = response.json()  # type: ignore[attr-defined]
        payload = payload or {}

        dataset_data = payload.get("dataset", {})
        dataset = Dataset.from_dict(dataset_data)

        # Log warning if present
        if warning := payload.get("warning"):
            self._logger.warning("Dataset API warning: %s", warning)

        return dataset

    def _dataset_items_url(self, slug: str) -> str:
        return f"{self._base_url}/datasets/{slug}/items"

    def _dataset_row_from_response(self, response: object) -> DatasetRow:
        # The response is expected to be an HTTP response object with a .json() method.
        response = self._ensure_response(response)
        if not hasattr(response, "json") or not callable(getattr(response, "json", None)):
            raise BasaltAPIError("Response object does not have a callable .json() method")
        payload = response.json()  # type: ignore[attr-defined]
        payload = payload or {}
        row_data = payload.get("datasetRow", {})

        # Log warning if present
        if warning := payload.get("warning"):
            self._logger.warning("Dataset API warning: %s", warning)

        return DatasetRow.from_dict(row_data)

    def _build_dataset_row_request(
        self,
        slug: str,
        processed_values: dict[str, str],
        name: str | None,
        ideal_output: str | None,
        metadata: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        url = self._dataset_items_url(slug)
        body: dict[str, Any] = {
            "values": processed_values,
        }
        if name is not None:
            body["name"] = name
        if ideal_output is not None:
            body["idealOutput"] = ideal_output
        if metadata is not None:
            body["metadata"] = metadata

        span_attributes = {
            "basalt.dataset.slug": slug,
            "basalt.dataset.row_name": name,
        }
        return url, body, span_attributes

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
        if not isinstance(datasets_data, list):
            return []
        return [Dataset.from_dict(ds) for ds in datasets_data if isinstance(ds, dict)]

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
        if not isinstance(datasets_data, list):
            return []
        return [Dataset.from_dict(ds) for ds in datasets_data if isinstance(ds, dict)]

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

        return self._dataset_from_response(response)

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

        return self._dataset_from_response(response)

    async def _process_file_uploads(
        self, values: dict[str, str | FileAttachment]
    ) -> dict[str, str]:
        """
        Process file uploads and return values with S3 keys.

        Args:
            values: Dictionary containing strings and/or FileAttachment objects

        Returns:
            Dictionary with all FileAttachments replaced by their S3 keys

        Raises:
            FileValidationError: If file validation fails
            FileUploadError: If file upload fails
        """
        processed = {}

        for key, value in values.items():
            if isinstance(value, FileAttachment):
                # Upload file and get S3 key
                s3_key = await self._file_upload_handler.upload_file(value)
                processed[key] = s3_key
            else:
                processed[key] = value

        return processed

    def _process_file_uploads_sync(self, values: dict[str, str | FileAttachment]) -> dict[str, str]:
        """
        Process file uploads and return values with S3 keys (synchronous version).

        Args:
            values: Dictionary containing strings and/or FileAttachment objects

        Returns:
            Dictionary with all FileAttachments replaced by their S3 keys

        Raises:
            FileValidationError: If file validation fails
            FileUploadError: If file upload fails
        """
        processed = {}

        for key, value in values.items():
            if isinstance(value, FileAttachment):
                # Upload file and get S3 key
                s3_key = self._file_upload_handler.upload_file_sync(value)
                processed[key] = s3_key
            else:
                processed[key] = value

        return processed

    async def add_row(
        self,
        slug: str,
        values: dict[str, str | FileAttachment],
        name: str | None = None,
        ideal_output: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetRow:
        """
        Create a new item (row) in a dataset.

        Args:
            slug: The slug identifier for the dataset.
            values: A dictionary of column values for the dataset item.
                Values can be strings or FileAttachment objects for file columns.
                Files are automatically uploaded to S3 before creating the row.
            name: An optional name for the dataset item.
            ideal_output: An optional ideal output for the dataset item.
            metadata: An optional metadata dictionary.

        Returns:
            The created DatasetRow. File values will contain S3 keys.

        Raises:
            FileValidationError: If file validation fails.
            FileUploadError: If file upload fails.
            BasaltAPIError: If the API request fails.
            NetworkError: If a network error occurs.
        """
        # Process file uploads first
        processed_values = await self._process_file_uploads(values)

        url, body, span_attributes = self._build_dataset_row_request(
            slug, processed_values, name, ideal_output, metadata
        )

        response = await self._request_async(
            "add_row",
            method="POST",
            url=url,
            body=body,
            headers=self._get_headers(),
            span_attributes=span_attributes,
        )

        return self._dataset_row_from_response(response)

    def add_row_sync(
        self,
        slug: str,
        values: dict[str, str | FileAttachment],
        name: str | None = None,
        ideal_output: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetRow:
        """
        Synchronously create a new item (row) in a dataset.

        Args:
            slug: The slug identifier for the dataset.
            values: A dictionary of column values for the dataset item.
                Values can be strings or FileAttachment objects for file columns.
                Files are automatically uploaded to S3 before creating the row.
            name: An optional name for the dataset item.
            ideal_output: An optional ideal output for the dataset item.
            metadata: An optional metadata dictionary.

        Returns:
            The created DatasetRow. File values will contain S3 keys.

        Raises:
            FileValidationError: If file validation fails.
            FileUploadError: If file upload fails.
            BasaltAPIError: If the API request fails.
            NetworkError: If a network error occurs.
        """
        # Process file uploads first
        processed_values = self._process_file_uploads_sync(values)

        url, body, span_attributes = self._build_dataset_row_request(
            slug, processed_values, name, ideal_output, metadata
        )

        response = self._request_sync(
            "add_row",
            method="POST",
            url=url,
            body=body,
            headers=self._get_headers(),
            span_attributes=span_attributes,
        )

        return self._dataset_row_from_response(response)

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
