from __future__ import annotations

import unittest

from basalt.client import Basalt
from basalt.experiments.client import ExperimentsClient


class TestBasaltClientExperiments(unittest.TestCase):
    def test_basalt_exposes_experiments_client(self) -> None:
        client = Basalt(api_key="key")
        self.assertIsInstance(client.experiments, ExperimentsClient)
        # Subsequent calls return same object
        self.assertIs(client.experiments, client.experiments)
        client.shutdown()


if __name__ == "__main__":
    unittest.main()
