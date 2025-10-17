"""Test deprecation warnings for old tuple-based API."""
from __future__ import annotations

import warnings
from unittest.mock import patch

import pytest

from basalt.basalt_facade import BasaltFacade


class TestDeprecationWarnings:
    """Test suite for deprecation warnings."""

    @patch("basalt.prompts.client.HTTPClient.fetch_sync")
    def test_prompt_get_emits_deprecation_warning(self, mock_fetch):
        """Test that using prompt.get_sync() emits a deprecation warning."""
        mock_fetch.return_value = {"prompt": {
            "text": "Hello",
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

        # Test that deprecation warning is emitted
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            err, prompt, generation = facade.prompt.get_sync("test")

            # Check warning was raised
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "tuple-based api" in str(w[0].message).lower()
            assert "v1.0.0" in str(w[0].message)
            assert "migration-guide" in str(w[0].message).lower()

    @patch("basalt.prompts.client.HTTPClient.fetch_sync")
    def test_warning_only_emitted_once_per_instance(self, mock_fetch):
        """Test that warning is only emitted once per adapter instance."""
        mock_fetch.return_value = {"prompt": {
            "text": "Hello",
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

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # First call - should warn
            facade.prompt.get_sync("test")
            assert len(w) == 1

            # Second call - should not warn again
            facade.prompt.get_sync("test")
            assert len(w) == 1  # Still only 1 warning

    @patch("basalt.datasets.client.HTTPClient.fetch_sync")
    def test_dataset_list_emits_deprecation_warning(self, mock_fetch):
        """Test that using datasets.list_sync() emits a deprecation warning."""
        mock_fetch.return_value = {"datasets": []}

        facade = BasaltFacade(api_key="test-key")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            err, datasets = facade.datasets.list_sync()

            # Check warning was raised
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "tuple-based api" in str(w[0].message).lower()
            assert "DatasetsClient" in str(w[0].message)

    @pytest.mark.asyncio
    @patch("basalt.prompts.client.HTTPClient.fetch")
    async def test_async_methods_also_emit_warnings(self, mock_fetch):
        """Test that async methods also emit deprecation warnings."""
        mock_fetch.return_value = {"prompt": {
            "text": "Hello",
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

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            err, prompt, generation = await facade.prompt.get("test")

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

    def test_warning_message_contains_useful_info(self):
        """Test that warning message contains actionable information."""
        facade = BasaltFacade(api_key="test-key")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            with patch("basalt.prompts.client.HTTPClient.fetch_sync") as mock_fetch:
                mock_fetch.return_value = {"prompt": {
                    "text": "Hello",
                    "slug": "test",
                    "version": "1.0.0",
                    "tag": "prod",
                    "systemText": "",
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

                facade.prompt.get_sync("test")

            message = str(w[0].message)

            # Check for key information
            assert "deprecated" in message.lower()
            assert "v1.0.0" in message  # Version when it will be removed
            assert "PromptsClient" in message  # What to use instead
            assert "exception" in message.lower()  # Error handling approach
            assert "migration-guide" in message.lower()  # Where to find help


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
