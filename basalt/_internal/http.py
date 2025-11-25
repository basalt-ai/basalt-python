"""HTTP client for the Basalt SDK."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from ..types.exceptions import (
    BadRequestError,
    ForbiddenError,
    NetworkError,
    NotFoundError,
    UnauthorizedError,
)

logger = logging.getLogger(__name__)

# Type alias for HTTP methods
HTTPMethod = Literal["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]

# Default configuration constants
DEFAULT_TIMEOUT = 30.0  # seconds
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_FACTOR = 0.5  # seconds


@dataclass(slots=True)
class HTTPResponse(Mapping[str, Any]):
    """Thin wrapper around an HTTP JSON payload with associated metadata."""

    status_code: int
    data: dict[str, Any] | None
    headers: Mapping[str, str]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data or {})

    def __len__(self) -> int:
        return len(self.data or {})

    def __getitem__(self, key: str) -> Any:
        if not self.data:
            raise KeyError(key)
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        if not self.data:
            return default
        return self.data.get(key, default)

    def json(self) -> dict[str, Any] | None:
        return self.data

    @property
    def body(self) -> dict[str, Any] | None:
        return self.data


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
        async_client: httpx.AsyncClient | None = None,
        sync_client: httpx.Client | None = None,
    ):
        """
        Initialize HTTPClient with configuration options.

        Args:
            timeout: Request timeout in seconds (default: 30.0)
            max_retries: Maximum number of retry attempts for transient errors (default: 3)
            verify_ssl: Whether to verify SSL certificates (default: True)
            retry_backoff_factor: Exponential backoff factor for retries (default: 0.5)
            async_client: Optional pre-configured httpx.AsyncClient instance.
            sync_client: Optional pre-configured httpx.Client instance.
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.verify_ssl = verify_ssl
        self.retry_backoff_factor = retry_backoff_factor

        self._async_client = async_client
        self._sync_client = sync_client
        self._owns_async_client = async_client is None
        self._owns_sync_client = sync_client is None

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
        if self._async_client and self._owns_async_client:
            await self._async_client.aclose()
            self._async_client = None

    def close(self):
        """Close the sync session."""
        if self._sync_client and self._owns_sync_client:
            self._sync_client.close()
            self._sync_client = None

    def _ensure_sync_client(self) -> httpx.Client:
        """Get or create a sync httpx client."""
        if self._sync_client is None:
            self._sync_client = httpx.Client(timeout=self.timeout, verify=self.verify_ssl)
        return self._sync_client

    def _ensure_async_client(self) -> httpx.AsyncClient:
        """Get or create an async httpx client."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self.timeout, verify=self.verify_ssl)
        return self._async_client

    async def fetch(
        self,
        url: str,
        method: str | HTTPMethod,
        body: Any | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HTTPResponse | None:
        """
        Asynchronously fetch data from a URL using httpx.

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

                client = self._ensure_async_client()
                response = await client.request(
                    method.upper(),
                    url,
                    params=filtered_params or None,
                    json=body,
                    headers=filtered_headers or None,
                    timeout=self.timeout,
                )
                return self._handle_response(response)

            except (BadRequestError, UnauthorizedError, ForbiddenError, NotFoundError):
                # Don't retry client errors
                raise
            except (httpx.TimeoutException, asyncio.TimeoutError, httpx.TransportError) as e:
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

    def fetch_sync(
        self,
        url: str,
        method: str | HTTPMethod,
        body: Any | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HTTPResponse | None:
        """
        Synchronously fetch data from a URL using httpx.

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

                client = self._ensure_sync_client()
                response = client.request(
                    method.upper(),
                    url,
                    params=filtered_params or None,
                    json=body,
                    headers=filtered_headers or None,
                    timeout=self.timeout,
                )
                return self._handle_response(response)

            except (BadRequestError, UnauthorizedError, ForbiddenError, NotFoundError):
                # Don't retry client errors
                raise
            except (httpx.TimeoutException, httpx.TransportError) as e:
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
    def _handle_response(response: httpx.Response) -> HTTPResponse | None:
        """
        Handles httpx response parsing and error raising.

        Returns:
            HTTPResponse when JSON content is present, or None for 204/empty body.
        """
        status = response.status_code
        headers_obj = getattr(response, "headers", {})
        if isinstance(headers_obj, Mapping):
            raw_content_type = (
                headers_obj.get("content-type")
                or headers_obj.get("Content-Type")
                or ""
            )
        else:
            raw_content_type = str(headers_obj or "")
        content_type = str(raw_content_type).lower()
        is_json = "json" in content_type
        json_response: dict[str, Any] | None = None
        text_response: str = ""

        # Handle error responses
        if status >= 400:
            if is_json:
                try:
                    json_response = response.json()
                except Exception:
                    json_response = None
                    text_response = response.text
                else:
                    text_response = response.text if json_response is None else ""
            else:
                text_response = response.text
            HTTPClient._raise_for_status(status, json_response, text_response)

        # Handle success responses
        if status in (202, 204):
            if response.content in (b"", None):
                return HTTPResponse(
                    status_code=status,
                    data=None,
                    headers=dict(response.headers),
                )
            if is_json:
                try:
                    json_response = response.json()
                except Exception as e:
                    logger.debug("Expected JSON response but failed to parse", exc_info=e)
                    return HTTPResponse(
                        status_code=status,
                        data=None,
                        headers=dict(response.headers),
                    )
                return HTTPResponse(
                    status_code=status,
                    data=json_response if json_response is not None else None,
                    headers=dict(response.headers),
                )
            else:
                return HTTPResponse(
                    status_code=status,
                    data=None,
                    headers=dict(response.headers),
                )
        elif status in (200, 201):
            # Accept valid JSON from mock even if content is b'{}'
            try:
                json_response = response.json()
                return HTTPResponse(
                    status_code=status,
                    data=json_response if json_response is not None else None,
                    headers=dict(response.headers),
                )
            except Exception as exc:
                raise NetworkError("Invalid JSON response") from exc
        elif status < 400:
            if response.content:
                if not is_json:
                    raise NetworkError("Expected JSON response")
                try:
                    json_response = response.json()
                except Exception as exc:
                    raise NetworkError("Invalid JSON response") from exc
            return HTTPResponse(
                status_code=status,
                data=json_response if json_response else None,
                headers=dict(response.headers),
            )
        return None

    @staticmethod
    def _raise_for_status(
        status: int,
        json_response: dict[str, Any] | None,
        text_response: str,
    ) -> None:
        """
        Raises custom exceptions for HTTP error codes.

        Extracts "error" or "errors" field from JSON responses when available.
        """
        # Extract error message from JSON response if available
        message = None
        if json_response:
            if "error" in json_response:
                message = json_response["error"]
            elif "errors" in json_response:
                message = json_response["errors"]

        # Fall back to text response or default message
        if message is None:
            message = text_response

        if status == 400:
            raise BadRequestError(message or "Bad Request")
        elif status == 401:
            raise UnauthorizedError(message or "Unauthorized")
        elif status == 403:
            raise ForbiddenError(message or "Forbidden")
        elif status == 404:
            raise NotFoundError(message or "Not Found")
        else:
            raise NetworkError(message or f"HTTP {status}")
