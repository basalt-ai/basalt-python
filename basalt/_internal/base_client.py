"""Shared helpers for Basalt service clients."""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Any

from basalt.observability.request_tracing import (
    trace_async_request,
    trace_sync_request,
)
from basalt.observability.spans import BasaltRequestSpan

from .http import HTTPClient


class BaseServiceClient:
    """Provide request execution helpers with consistent tracing behaviour."""

    def __init__(self, *, client_name: str, http_client: HTTPClient | None = None) -> None:
        self._client_name = client_name
        self._http_client = http_client or HTTPClient()

    @staticmethod
    def _filter_attributes(attributes: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if attributes is None:
            return None
        filtered = {key: value for key, value in attributes.items() if value is not None}
        return filtered or None

    async def _request_async(
        self,
        operation: str,
        *,
        method: str,
        url: str,
        span_attributes: Mapping[str, Any] | None = None,
        cache_hit: bool | None = None,
        **request_kwargs: Any,
    ):
        span = BasaltRequestSpan(
            client=self._client_name,
            operation=operation,
            method=method,
            url=url,
            cache_hit=cache_hit,
            extra_attributes=self._filter_attributes(span_attributes),
        )
        call = functools.partial(
            self._http_client.fetch,
            url=url,
            method=method,
            **request_kwargs,
        )
        return await trace_async_request(span, call)

    def _request_sync(
        self,
        operation: str,
        *,
        method: str,
        url: str,
        span_attributes: Mapping[str, Any] | None = None,
        cache_hit: bool | None = None,
        **request_kwargs: Any,
    ):
        span = BasaltRequestSpan(
            client=self._client_name,
            operation=operation,
            method=method,
            url=url,
            cache_hit=cache_hit,
            extra_attributes=self._filter_attributes(span_attributes),
        )
        call = functools.partial(
            self._http_client.fetch_sync,
            url=url,
            method=method,
            **request_kwargs,
        )
        return trace_sync_request(span, call)
