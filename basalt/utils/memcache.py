import time
from collections.abc import Hashable
from typing import TypeVar

from ..types.cache import CacheProtocol

T = TypeVar("T")


class MemoryCache(CacheProtocol):
    """
    MemoryCache is a simple in-memory cache that stores values for a given key.
    It implements the ICache protocol.
    """

    def __init__(self) -> None:
        self._mem: dict[Hashable, object] = {}
        self._timeouts: dict[Hashable, float] = {}

    def get(self, key: Hashable) -> object | None:
        """
        Retrieves the value associated with the given key if it has not expired.

        Args:
            key (Hashable): The key to look up in the cache.

        Returns:
            The value associated with the key, or None if the key does not exist or has expired.
        """
        mem = self._mem.get(key)
        timeout = self._timeouts.get(key)

        if timeout is None or timeout > time.time():
            return mem

        return None

    def put(self, key: Hashable, value: object, ttl: float = float("inf")) -> None:
        """
        Stores a value in the cache with an associated time-to-live (TTL).

        Args:
            key (Hashable): The key to associate with the value.
            value (Any): The value to store in the cache.
            ttl (float, optional): The time-to-live for the cache entry in seconds. Defaults to infinity.
        """
        self._mem[key] = value
        self._timeouts[key] = time.time() + ttl
