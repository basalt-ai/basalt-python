"""HTTP client for the Basalt SDK."""

import asyncio
import time
from collections.abc import Mapping
from typing import Any, Literal

import aiohttp
import requests

from basalt.observability.decorators import trace_http

from ..types.exceptions import (
    BadRequestError,
    ForbiddenError,
    NetworkError,
    NotFoundError,
    UnauthorizedError,
)

# Type alias for HTTP methods
HTTPMethod = Literal["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]

# Default configuration constants
DEFAULT_TIMEOUT = 30.0  # seconds
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_FACTOR = 0.5  # seconds


class HTTPClient:
    """
    HTTP client for making requests to the Basalt API.

    Provides synchronous and asynchronous methods that raise exceptions on HTTP errors.
    Supports session reuse, timeouts, retries, and SSL verification.
    """

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        verify_ssl: bool = True,
        retry_backoff_factor: float = DEFAULT_RETRY_BACKOFF_FACTOR,
    ):
        """
        Initialize HTTPClient with configuration options.

        Args:
            timeout: Request timeout in seconds (default: 30.0)
            max_retries: Maximum number of retry attempts for transient errors (default: 3)
            verify_ssl: Whether to verify SSL certificates (default: True)
            retry_backoff_factor: Exponential backoff factor for retries (default: 0.5)
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.verify_ssl = verify_ssl
        self.retry_backoff_factor = retry_backoff_factor
        self._session: aiohttp.ClientSession | None = None
        self._sync_session: requests.Session | None = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.aclose()

    def __enter__(self):
        """Sync context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context manager exit."""
        self.close()

    async def aclose(self):
        """Close the async session."""
        if self._session:
            await self._session.close()
            self._session = None

    def close(self):
        """Close the sync session."""
        if self._sync_session:
            self._sync_session.close()
            self._sync_session = None

    def _get_sync_session(self) -> requests.Session:
        """Get or create a sync session."""
        if self._sync_session is None:
            self._sync_session = requests.Session()
        return self._sync_session

    async def _get_async_session(self) -> aiohttp.ClientSession:
        """Get or create an async session."""
        if self._session is None:
            connector = aiohttp.TCPConnector(ssl=self.verify_ssl)
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self._session

    @trace_http(name="basalt.http.fetch_async")
    async def fetch(
        self,
        url: str,
        method: str | HTTPMethod,
        body: Any | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any] | None:
        """
        Asynchronously fetch data from a URL using aiohttp.

        Args:
            url: The URL to fetch
            method: HTTP method (GET, POST, PUT, etc.)
            body: Request body (will be JSON-encoded)
            params: Query parameters
            headers: Request headers

        Returns:
            JSON response as dict, or None for 204 No Content

        Raises:
            BadRequestError: For 400 responses
            UnauthorizedError: For 401 responses
            ForbiddenError: For 403 responses
            NotFoundError: For 404 responses
            NetworkError: For network errors and other HTTP errors
        """
        for attempt in range(self.max_retries):
            try:
                filtered_params = {k: v for k, v in (params or {}).items() if v is not None}
                filtered_headers = {k: v for k, v in (headers or {}).items() if v is not None}

                session = await self._get_async_session()
                async with session.request(
                    method.upper(),
                    url,
                    params=filtered_params,
                    json=body,
                    headers=filtered_headers,
                ) as response:
                    result = await self._handle_response(response)
                    return result

            except (BadRequestError, UnauthorizedError, ForbiddenError, NotFoundError):
                # Don't retry client errors
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                # Retry on transient errors
                if attempt == self.max_retries - 1:
                    raise NetworkError(f"Request failed after {self.max_retries} attempts: {e}") from e

                # Exponential backoff
                wait_time = self.retry_backoff_factor * (2 ** attempt)
                await asyncio.sleep(wait_time)
            except Exception as e:
                raise NetworkError(str(e)) from e

        # Should never reach here, but just in case
        raise NetworkError(f"Request failed after {self.max_retries} attempts")

    @trace_http(name="basalt.http.fetch_sync")
    def fetch_sync(
        self,
        url: str,
        method: str | HTTPMethod,
        body: Any | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any] | None:
        """
        Synchronously fetch data from a URL using requests.

        Args:
            url: The URL to fetch
            method: HTTP method (GET, POST, PUT, etc.)
            body: Request body (will be JSON-encoded)
            params: Query parameters
            headers: Request headers

        Returns:
            JSON response as dict, or None for 204 No Content

        Raises:
            BadRequestError: For 400 responses
            UnauthorizedError: For 401 responses
            ForbiddenError: For 403 responses
            NotFoundError: For 404 responses
            NetworkError: For network errors and other HTTP errors
        """
        for attempt in range(self.max_retries):
            try:
                session = self._get_sync_session()
                response = session.request(
                    method.upper(),
                    url,
                    params=params,
                    json=body,
                    headers=headers,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                )
                result = self._handle_sync_response(response)
                return result

            except (BadRequestError, UnauthorizedError, ForbiddenError, NotFoundError):
                # Don't retry client errors
                raise
            except (requests.RequestException, requests.Timeout) as e:
                # Retry on transient errors
                if attempt == self.max_retries - 1:
                    raise NetworkError(f"Request failed after {self.max_retries} attempts: {e}") from e

                # Exponential backoff
                wait_time = self.retry_backoff_factor * (2 ** attempt)
                time.sleep(wait_time)
            except Exception as e:
                raise NetworkError(str(e)) from e

        # Should never reach here, but just in case
        raise NetworkError(f"Request failed after {self.max_retries} attempts")

    @staticmethod
    async def _handle_response(response: aiohttp.ClientResponse) -> dict[str, Any] | None:
        """
        Handles aiohttp response parsing and error raising.

        Returns:
            JSON response as dict, or None for 204 No Content
        """
        content_type = response.headers.get('Content-Type', '')
        json_response: dict[str, Any] = {}
        text_response: str = ""

        # For error responses, try to get the response body (JSON or text)
        if response.status >= 400:
            if 'application/json' in content_type:
                try:
                    json_response = await response.json()
                except Exception:
                    # Fall back to text if JSON parsing fails
                    text_response = await response.text()
            else:
                text_response = await response.text()

            HTTPClient._raise_for_status(response.status, json_response, text_response)

        # For successful responses
        if response.status == 204:
            # No Content
            return None
        elif response.status in [200, 201, 202]:
            if 'application/json' in content_type:
                try:
                    json_response = await response.json()
                except Exception as exc:
                    raise NetworkError("Invalid JSON response") from exc
            elif response.status != 202:
                # 202 Accepted can have no content
                raise NetworkError("Expected JSON response")

        return json_response if json_response else None

    @staticmethod
    def _handle_sync_response(response: requests.Response) -> dict[str, Any] | None:
        """
        Handles requests response parsing and error raising.

        Returns:
            JSON response as dict, or None for 204 No Content
        """
        content_type = response.headers.get('Content-Type', '')
        json_response: dict[str, Any] = {}
        text_response: str = ""

        # For error responses, try to get the response body (JSON or text)
        if response.status_code >= 400:
            if 'application/json' in content_type:
                try:
                    json_response = response.json()
                except Exception:
                    # Fall back to text if JSON parsing fails
                    text_response = response.text
            else:
                text_response = response.text

            HTTPClient._raise_for_status(response.status_code, json_response, text_response)

        # For successful responses
        if response.status_code == 204:
            # No Content
            return None
        elif response.status_code in [200, 201, 202]:
            if 'application/json' in content_type:
                try:
                    json_response = response.json()
                except Exception as exc:
                    raise NetworkError("Invalid JSON response") from exc
            elif response.status_code != 202:
                # 202 Accepted can have no content
                raise NetworkError("Expected JSON response")

        return json_response if json_response else None

    @staticmethod
    def _raise_for_status(status: int, json_data: dict[str, Any], text_data: str = "") -> None:
        """
        Raises appropriate exception based on HTTP status code.

        Args:
            status: HTTP status code
            json_data: Parsed JSON response (if available)
            text_data: Raw text response (fallback if JSON not available)
        """
        # Try to extract error message from JSON first, then fall back to text
        error_msg = json_data.get('error') or json_data.get('errors') or text_data or "Unknown Error"

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
