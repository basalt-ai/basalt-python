"""Tests for telemetry configuration helpers."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from basalt.observability.config import OpenLLMetryConfig, TelemetryConfig


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

    def test_resolved_openllmetry_applies_defaults(self):
        config = TelemetryConfig(enable_openllmetry=True)

        resolved = config.resolved_openllmetry()

        self.assertIsNotNone(resolved)
        if resolved is None:  # pragma: no cover - guard for type checkers
            return
        self.assertEqual(resolved.app_name, config.service_name)


class TestOpenLLMetryConfig(unittest.TestCase):
    def test_clone_returns_independent_copy(self):
        original = OpenLLMetryConfig(
            headers={"Authorization": "secret"},
        )

        clone = original.clone()
        if clone.headers:
            clone.headers["Authorization"] = "changed"

        self.assertEqual(original.headers, {"Authorization": "secret"})
