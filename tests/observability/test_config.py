"""Tests for telemetry configuration helpers."""

from __future__ import annotations

from basalt.observability.config import TelemetryConfig


def test_env_overrides_respect_environment_variables(monkeypatch):
    monkeypatch.setenv("BASALT_TELEMETRY_ENABLED", "0")
    monkeypatch.setenv("BASALT_SERVICE_NAME", "env-service")
    monkeypatch.setenv("BASALT_ENVIRONMENT", "staging")

    config = TelemetryConfig(service_name="sdk").with_env_overrides()

    assert not config.enabled
    assert config.service_name == "env-service"
    assert config.environment == "staging"


def test_clone_returns_independent_copy_of_provider_lists():
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
    assert original.enabled_providers == ["openai", "anthropic"]
    assert original.disabled_providers == ["langchain"]
