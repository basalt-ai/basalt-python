"""
Experiments API Client.

This module provides the ExperimentsClient for interacting with the Basalt Experiments API.
"""
from __future__ import annotations

from typing import Any

from .._internal.exceptions import BasaltAPIError
from .._internal.http import HTTPClient
from ..config import config
from .models import Experiment


class ExperimentsClient:
    """
    Client for interacting with the Basalt Experiments API.

    This client provides methods to create experiments.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        http_client: HTTPClient | None = None,
    ):
        """
        Initialize the ExperimentsClient.

        Args:
            api_key: The Basalt API key for authentication.
            base_url: Optional base URL for the API (defaults to config value).
            http_client: Optional HTTP client instance for making requests.
        """
        self._api_key = api_key
        self._base_url = base_url or config["api_url"]
        self._http_client = http_client or HTTPClient()

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

        response = await self._http_client.fetch(
            url=url,
            method="POST",
            body=body,
            headers=self._get_headers(),
        )

        response = response or {}
        if response.get("error"):
            raise BasaltAPIError(response["error"])

        return Experiment.from_dict(response)

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

        response = self._http_client.fetch_sync(
            url=url,
            method="POST",
            body=body,
            headers=self._get_headers(),
        )

        response = response or {}
        if response.get("error"):
            raise BasaltAPIError(response["error"])

        return Experiment.from_dict(response)

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
