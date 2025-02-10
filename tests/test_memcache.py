import unittest

from unittest.mock import MagicMock, patch
from parameterized import parameterized
from dataclasses import dataclass
from basalt.utils.memcache import MemoryCache

time_mock = MagicMock()

# A dataclass is always hashable
@dataclass(frozen=True)
class SomeHashableDTO:
	name: str
	age: int

class TestMemCache(unittest.TestCase):
	@parameterized.expand([
		("key", "value"),
		("key2", 123),
		("key3", { "bar": 1 }),
		("key4", ["a", "b", "c"]),
	])
	def test_can_put_with_str_key(self, key, val):
		cache = MemoryCache()
		cache.put(key, val)

		self.assertEqual(cache.get(key), val)

	def test_can_use_hashable_as_key(self):
		key = SomeHashableDTO(name="John", age=30)

		cache = MemoryCache()
		cache.put(key, 1)

		self.assertEqual(cache.get(key), 1)

	@patch("time.time")
	def test_cache_times_out_after_ttl(self, time_mock):
		time_mock.return_value = 0.0

		cache = MemoryCache()
		cache.put("abc123", "value", ttl=200.0)

		time_mock.return_value = 201.0

		self.assertIsNone(cache.get("abc123"))