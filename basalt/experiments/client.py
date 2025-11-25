"""
Experiments API Client.

This module provides the ExperimentsClient for interacting with the Basalt Experiments API.
"""
from __future__ import annotations

from typing import Any

from .._internal.base_client import BaseServiceClient
from .._internal.http import HTTPClient
from ..config import config
from ..types.exceptions import BasaltAPIError
from .models import Experiment


class ExperimentsClient(BaseServiceClient):
    """
    Client for interacting with the Basalt Experiments API.

    This client provides methods to create experiments.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        http_client: HTTPClient | None = None,
        log_level: str | None = None,
    ):
        """
        Initialize the ExperimentsClient.

        Args:
            api_key: The Basalt API key for authentication.
            base_url: Optional base URL for the API (defaults to config value).
            http_client: Optional HTTP client instance for making requests.
            log_level: Optional log level for the client logger.
        """
        self._api_key = api_key
        self._base_url = base_url or config.get("api_url")
        super().__init__(client_name="experiments", http_client=http_client, log_level=log_level)

    async def create(
        self,
        feature_slug: str,
        name: str,
    ) -> Experiment:
        """
        Create a new experiment.

        Args:
            feature_slug: The feature slug to associate with the experiment.
            name: The name of the experiment.

        Returns:
            An Experiment object containing the created experiment data.

        Raises:
            BasaltAPIError: If the API request fails.
            NetworkError: If a network error occurs.
        """
        url = f"{self._base_url}/monitor/experiments"

        body: dict[str, Any] = {
            "featureSlug": feature_slug,
            "name": name,
        }

        response = await self._request_async(
            "create",
            method="POST",
            url=url,
            body=body,
            headers=self._get_headers(),
            span_attributes={
                "basalt.experiment.feature_slug": feature_slug,
                "basalt.experiment.name": name,
            },
        )

        if response is None:
            raise BasaltAPIError("Empty response from experiment API")

        payload = response.json() or {}

        return Experiment.from_dict(payload)

    def create_sync(
        self,
        feature_slug: str,
        name: str,
    ) -> Experiment:
        """
        Synchronously create a new experiment.

        Args:
            feature_slug: The feature slug to associate with the experiment.
            name: The name of the experiment.

        Returns:
            An Experiment object containing the created experiment data.

        Raises:
            BasaltAPIError: If the API request fails.
            NetworkError: If a network error occurs.
        """
        url = f"{self._base_url}/monitor/experiments"

        body: dict[str, Any] = {
            "featureSlug": feature_slug,
            "name": name,
        }

        response = self._request_sync(
            "create",
            method="POST",
            url=url,
            body=body,
            headers=self._get_headers(),
            span_attributes={
                "basalt.experiment.feature_slug": feature_slug,
                "basalt.experiment.name": name,
            },
        )

        if response is None:
            raise BasaltAPIError("Empty response from experiment API")

        payload = response.json() or {}

        return Experiment.from_dict(payload)

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
