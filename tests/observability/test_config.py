"""Tests for telemetry configuration helpers."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from basalt.observability.config import TelemetryConfig


class TestTelemetryConfig(unittest.TestCase):
    def test_env_overrides_respect_environment_variables(self):
        with mock.patch.dict(
            os.environ,
            {
                "BASALT_TELEMETRY_ENABLED": "0",
                "BASALT_SERVICE_NAME": "env-service",
                "BASALT_ENVIRONMENT": "staging",
            },
            clear=False,
        ):
            config = TelemetryConfig(service_name="sdk").with_env_overrides()

        self.assertFalse(config.enabled)
        self.assertEqual(config.service_name, "env-service")
        self.assertEqual(config.environment, "staging")

    def test_clone_returns_independent_copy_of_provider_lists(self):
        """Test that cloning creates independent copies of provider lists."""
        original = TelemetryConfig(
            enabled_providers=["openai", "anthropic"],
            disabled_providers=["langchain"],
        )

        clone = original.clone()
        if clone.enabled_providers:
            clone.enabled_providers.append("cohere")
        if clone.disabled_providers:
            clone.disabled_providers.append("llamaindex")

        # Original should be unchanged
        self.assertEqual(original.enabled_providers, ["openai", "anthropic"])
        self.assertEqual(original.disabled_providers, ["langchain"])
