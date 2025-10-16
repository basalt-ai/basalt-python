"""HTTP client for the Basalt SDK."""

from collections.abc import Mapping
from typing import Any

import aiohttp
import requests

from .exceptions import (
    BadRequestError,
    ForbiddenError,
    NetworkError,
    NotFoundError,
    UnauthorizedError,
)


class HTTPClient:
    """
    HTTP client for making requests to the Basalt API.

    Provides synchronous and asynchronous methods that raise exceptions on HTTP errors.
    """

    def __init__(self):
        pass  # Constructor placeholder, kept in case of future instance-level setup.

    @staticmethod
    async def fetch(
        url: str,
        method: str,
        body: Any | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Asynchronously fetch data from a URL using aiohttp."""
        try:
            filtered_params = {k: v for k, v in (params or {}).items() if v is not None}
            filtered_headers = {k: v for k, v in (headers or {}).items() if v is not None}

            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
                async with session.request(
                    method.upper(),
                    url,
                    params=filtered_params,
                    json=body,
                    headers=filtered_headers,
                ) as response:
                    return await HTTPClient._handle_response(response)

        except (BadRequestError, UnauthorizedError, ForbiddenError, NotFoundError):
            raise  # Preserve traceback for known errors
        except Exception as e:
            raise NetworkError(str(e)) from e

    @staticmethod
    def fetch_sync(
        url: str,
        method: str,
        body: Any | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Synchronously fetch data from a URL using requests."""
        try:
            response = requests.request(
                method.upper(),
                url,
                params=params,
                json=body,
                headers=headers,
            )
            return HTTPClient._handle_sync_response(response)

        except (BadRequestError, UnauthorizedError, ForbiddenError, NotFoundError):
            raise
        except Exception as e:
            raise NetworkError(str(e)) from e

    @staticmethod
    async def _handle_response(response: aiohttp.ClientResponse) -> dict[str, Any]:
        """Handles aiohttp response parsing and error raising."""
        content_type = response.headers.get('Content-Type', '')
        json_response: dict[str, Any] = {}

        if 'application/json' in content_type:
            try:
                json_response = await response.json()
            except Exception as exc:
                raise NetworkError("Invalid JSON response") from exc
        elif response.status not in [202, 204]:
            raise NetworkError("Expected JSON response")

        HTTPClient._raise_for_status(response.status, json_response)
        return json_response

    @staticmethod
    def _handle_sync_response(response: requests.Response) -> dict[str, Any]:
        """Handles requests response parsing and error raising."""
        content_type = response.headers.get('Content-Type', '')
        json_response: dict[str, Any] = {}

        if 'application/json' in content_type:
            try:
                json_response = response.json()
            except Exception as exc:
                raise NetworkError("Invalid JSON response") from exc
        elif response.status_code not in [202, 204]:
            raise NetworkError("Expected JSON response")

        HTTPClient._raise_for_status(response.status_code, json_response)
        return json_response

    @staticmethod
    def _raise_for_status(status: int, json_data: dict[str, Any]) -> None:
        """Raises appropriate exception based on HTTP status code."""
        error_msg = json_data.get('error') or json_data.get('errors') or "Unknown Error"

        match status:
            case 400:
                raise BadRequestError(error_msg)
            case 401:
                raise UnauthorizedError(error_msg)
            case 403:
                raise ForbiddenError(error_msg)
            case 404:
                raise NotFoundError(error_msg)
            case _ if status >= 400:
                raise NetworkError(f"HTTP {status}: {error_msg}")
