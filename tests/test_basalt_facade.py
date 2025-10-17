"""Tests for BasaltFacade with new clients."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from basalt.basalt_facade import BasaltFacade


class TestBasaltFacadeIntegration:
    """Test BasaltFacade integration with new clients."""

    def test_facade_initializes_with_new_clients(self):
        """Test that BasaltFacade initializes successfully."""
        facade = BasaltFacade(api_key="test-key")

        assert facade.prompt is not None
        assert facade.datasets is not None
        assert facade.monitor is not None

    def test_facade_rejects_empty_api_key(self):
        """Test that empty API key raises ValueError."""
        with pytest.raises(ValueError, match="API key cannot be empty"):
            BasaltFacade(api_key="")

        with pytest.raises(ValueError, match="API key cannot be empty"):
            BasaltFacade(api_key="   ")

    @patch("basalt.prompts.client.HTTPClient.fetch_sync")
    def test_prompt_get_through_facade(self, mock_fetch):
        """Test prompt.get() works through facade."""
        mock_fetch.return_value = {"prompt": {
            "text": "Hello World",
            "slug": "test",
            "version": "1.0.0",
            "tag": "prod",
            "systemText": "System",
            "model": {
                "provider": "openai",
                "model": "gpt-4",
                "version": "latest",
                "parameters": {
                    "temperature": 0.7,
                    "maxLength": 4096,
                    "responseFormat": "text",
                },
            },
        }}

        facade = BasaltFacade(api_key="test-key")
        err, prompt, generation = facade.prompt.get_sync("test")

        assert err is None
        assert prompt is not None
        assert prompt.slug == "test"
        assert generation is not None

    @patch("basalt.datasets.client.HTTPClient.fetch_sync")
    def test_datasets_list_through_facade(self, mock_fetch):
        """Test datasets.list() works through facade."""
        mock_fetch.return_value = {"datasets": [
            {"slug": "ds1", "name": "Dataset 1", "columns": ["col1"]},
        ]}

        facade = BasaltFacade(api_key="test-key")
        err, datasets = facade.datasets.list_sync()

        assert err is None
        assert datasets is not None
        assert len(datasets) == 1
        assert datasets[0].slug == "ds1"

    @patch("basalt.prompts.client.HTTPClient.fetch_sync")
    def test_error_handling_through_facade(self, mock_fetch):
        """Test error handling works through facade."""
        from basalt._internal.exceptions import NotFoundError

        mock_fetch.side_effect = NotFoundError("Not found")

        facade = BasaltFacade(api_key="test-key")
        err, prompt, generation = facade.prompt.get_sync("nonexistent")

        assert err is not None
        assert isinstance(err, NotFoundError)
        assert prompt is None
        assert generation is None

    def test_custom_cache_support(self):
        """Test facade accepts custom cache."""
        custom_cache = MagicMock()
        facade = BasaltFacade(api_key="test-key", cache=custom_cache)

        assert facade.prompt is not None

    def test_log_level_configuration(self):
        """Test facade accepts log level configuration."""
        facade = BasaltFacade(api_key="test-key", log_level="none")
        assert facade.prompt is not None

        facade = BasaltFacade(api_key="test-key", log_level="warning")
        assert facade.prompt is not None

        facade = BasaltFacade(api_key="test-key", log_level="all")
        assert facade.prompt is not None
