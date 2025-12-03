"""Typed span formats used across the Basalt SDK."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from . import semconv
from .context_managers import SpanHandle


@dataclass(slots=True)
class BasaltRequestSpan:
    """
    Typed span metadata shared by Basalt API clients.

    Attributes:
        client: Logical client name (e.g. ``"prompts"``).
        operation: Operation name within the client (e.g. ``"get"``).
        method: HTTP verb for the request.
        url: Fully-qualified request URL.
        cache_hit: Whether the request served from cache (if known).
        extra_attributes: Additional span attributes to attach at span creation.
        variables: Request variables payload to record on the span (optional).
    """

    client: str
    operation: str
    method: str
    url: str
    cache_hit: bool | None = None
    extra_attributes: Mapping[str, Any] | None = None
    variables: Mapping[str, Any] | None = None

    def span_name(self) -> str:
        return f"basalt.sdk.{self.client}.{self.operation}"

    def start_attributes(self) -> dict[str, Any]:
        attributes: dict[str, Any] = {
            semconv.BasaltAPI.CLIENT: self.client,
            semconv.BasaltAPI.OPERATION: self.operation,
            semconv.BasaltAPI.INTERNAL: True,
            semconv.HTTP.METHOD: self.method.upper(),
            semconv.HTTP.URL: self.url,
        }
        if self.cache_hit is not None:
            attributes[semconv.BasaltCache.HIT] = self.cache_hit
        if self.extra_attributes:
            for key, value in self.extra_attributes.items():
                if value is not None:
                    attributes[key] = value
        return attributes

    def finalize(
        self,
        span: SpanHandle,
        *,
        duration_s: float,
        status_code: int | None,
        error: BaseException | None = None,
    ) -> None:
        """Apply final attributes and status once the request completes."""
        duration_ms = round(duration_s * 1000, 2)
        span.set_attribute(semconv.BasaltRequest.DURATION_MS, duration_ms)
        span.set_attribute(semconv.HTTP.RESPONSE_TIME_MS, duration_ms)
        if status_code is not None:
            span.set_attribute(semconv.HTTP.STATUS_CODE, status_code)

        succeeded = error is None and (status_code is None or status_code < 400)
        span.set_attribute(semconv.BasaltRequest.SUCCESS, succeeded)
