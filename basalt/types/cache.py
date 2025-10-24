"""Cache protocol used throughout the Basalt SDK."""
from __future__ import annotations

from collections.abc import Hashable
from typing import Any, Protocol


class CacheProtocol(Protocol):
    """Minimal protocol implemented by cache backends."""

    def get(self, key: Hashable) -> Any | None:
        """Return a cached value for *key* or ``None`` when missing."""

    def put(self, key: Hashable, value: Any, ttl: float = float("inf")) -> None:
        """Store *value* for *key* with a time-to-live in seconds."""
