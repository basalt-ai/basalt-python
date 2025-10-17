"""
Integration tests for the PromptsClient.

These tests make real API requests to the Basalt backend. They can be toggled
using the BASALT_RUN_INTEGRATION_TESTS environment variable.

To run integration tests:
    export BASALT_RUN_INTEGRATION_TESTS=1
    export BASALT_API_KEY=your-api-key
    export BASALT_TEST_PROMPT_SLUG=your-test-prompt-slug
    python -m pytest tests/prompts/test_integration.py
"""
import os
import unittest

from basalt._internal.exceptions import NotFoundError
from basalt.prompts.client import PromptsClient
from basalt.prompts.models import DescribePromptResponse, Prompt, PromptListResponse
from basalt.utils.logger import Logger
from basalt.utils.memcache import MemoryCache


@unittest.skipUnless(
    os.getenv("BASALT_RUN_INTEGRATION_TESTS") == "1",
    "Integration tests disabled. Set BASALT_RUN_INTEGRATION_TESTS=1 to enable."
)
class TestPromptsClientIntegration(unittest.TestCase):
    """Integration test suite for PromptsClient."""

    @classmethod
    def setUpClass(cls):
        """Set up integration test fixtures."""
        cls.api_key = os.getenv("BASALT_API_KEY")
        if not cls.api_key:
            raise unittest.SkipTest("BASALT_API_KEY environment variable not set")

        cls.test_prompt_slug = os.getenv("BASALT_TEST_PROMPT_SLUG")
        if not cls.test_prompt_slug:
            raise unittest.SkipTest("BASALT_TEST_PROMPT_SLUG environment variable not set")

        cls.logger = Logger(log_level="all")
        cls.cache = MemoryCache()
        cls.fallback_cache = MemoryCache()

        cls.client = PromptsClient(
            api_key=cls.api_key,
            cache=cls.cache,
            fallback_cache=cls.fallback_cache,
            logger=cls.logger,
        )

    def test_get_sync_real_api(self):
        """Test synchronous prompt retrieval with real API."""
        prompt, generation = self.client.get_sync(self.test_prompt_slug)

        # Verify prompt object
        self.assertIsInstance(prompt, Prompt)
        self.assertEqual(prompt.slug, self.test_prompt_slug)
        self.assertIsNotNone(prompt.text)
        self.assertIsNotNone(prompt.model)
        self.assertIsNotNone(prompt.version)

        # Verify generation object
        self.assertIsNotNone(generation)
        self.assertEqual(generation.prompt["slug"], self.test_prompt_slug)

    def test_get_sync_with_version_real_api(self):
        """Test prompt retrieval with specific version."""
        # First describe to get available versions
        describe_response = self.client.describe_sync(self.test_prompt_slug)

        if describe_response.available_versions:
            version = describe_response.available_versions[0]
            prompt, generation = self.client.get_sync(
                self.test_prompt_slug,
                version=version
            )

            self.assertEqual(prompt.version, version)

    def test_get_sync_with_tag_real_api(self):
        """Test prompt retrieval with tag."""
        # First describe to get available tags
        describe_response = self.client.describe_sync(self.test_prompt_slug)

        if describe_response.available_tags:
            tag = describe_response.available_tags[0]
            prompt, generation = self.client.get_sync(
                self.test_prompt_slug,
                tag=tag
            )

            self.assertEqual(prompt.tag, tag)

    def test_get_sync_with_variables_real_api(self):
        """Test prompt retrieval with variable substitution."""
        # Get prompt metadata to check for variables
        describe_response = self.client.describe_sync(self.test_prompt_slug)

        if describe_response.variables:
            # Create variables dict from available variables
            variables = {
                var["name"]: f"test_{var['name']}"
                for var in describe_response.variables[:3]  # Use first 3 variables
            }

            prompt, generation = self.client.get_sync(
                self.test_prompt_slug,
                variables=variables
            )

            # Verify variables were applied
            self.assertIsNotNone(prompt.variables)
            for _key, value in variables.items():
                self.assertIn(value, prompt.text)

    def test_get_sync_cache_works_real_api(self):
        """Test that caching works with real API."""
        # Clear cache first
        self.cache = MemoryCache()
        self.client._cache = self.cache

        # First request - should hit API
        prompt1, _ = self.client.get_sync(self.test_prompt_slug)

        # Verify cache was populated
        cache_key = (self.test_prompt_slug, None, None)
        cached_value = self.cache.get(cache_key)
        self.assertIsNotNone(cached_value)

        # Second request - should use cache
        prompt2, _ = self.client.get_sync(self.test_prompt_slug)

        # Verify both prompts are the same
        self.assertEqual(prompt1.slug, prompt2.slug)
        self.assertEqual(prompt1.text, prompt2.text)

    def test_get_sync_not_found_real_api(self):
        """Test 404 error handling with real API."""
        with self.assertRaises(NotFoundError):
            self.client.get_sync("nonexistent-prompt-slug-12345")

    def test_describe_sync_real_api(self):
        """Test describe method with real API."""
        response = self.client.describe_sync(self.test_prompt_slug)

        # Verify response
        self.assertIsInstance(response, DescribePromptResponse)
        self.assertEqual(response.slug, self.test_prompt_slug)
        self.assertIsNotNone(response.name)
        self.assertIsNotNone(response.status)
        self.assertIsInstance(response.available_versions, list)
        self.assertIsInstance(response.available_tags, list)
        self.assertIsInstance(response.variables, list)

    def test_list_sync_real_api(self):
        """Test list method with real API."""
        prompts = self.client.list_sync()

        # Verify response
        self.assertIsInstance(prompts, list)
        if prompts:  # If there are prompts
            self.assertIsInstance(prompts[0], PromptListResponse)
            self.assertIsNotNone(prompts[0].slug)
            self.assertIsNotNone(prompts[0].name)
            self.assertIsNotNone(prompts[0].status)

    def test_fallback_cache_real_api(self):
        """Test fallback cache with real API."""
        # First, populate the fallback cache with a successful request
        prompt1, _ = self.client.get_sync(self.test_prompt_slug)

        # Now use an invalid API key to force an error
        bad_client = PromptsClient(
            api_key="invalid-key",
            cache=MemoryCache(),  # Empty primary cache
            fallback_cache=self.client._fallback_cache,  # Use populated fallback cache
            logger=self.logger,
        )

        # This should use the fallback cache instead of failing
        prompt2, _ = bad_client.get_sync(self.test_prompt_slug)

        # Verify we got the cached prompt
        self.assertEqual(prompt1.slug, prompt2.slug)
        self.assertEqual(prompt1.text, prompt2.text)

    def test_cache_disabled_real_api(self):
        """Test that cache can be disabled."""
        # Populate cache
        self.client.get_sync(self.test_prompt_slug)

        # Clear cache counters
        self.cache = MemoryCache()
        self.client._cache = self.cache

        # Request with cache disabled should still work
        prompt, _ = self.client.get_sync(self.test_prompt_slug, cache_enabled=False)

        # Verify cache was not used
        cache_key = (self.test_prompt_slug, None, None)
        self.assertIsNone(self.cache.get(cache_key))


@unittest.skipUnless(
    os.getenv("BASALT_RUN_INTEGRATION_TESTS") == "1",
    "Integration tests disabled. Set BASALT_RUN_INTEGRATION_TESTS=1 to enable."
)
class TestPromptsClientIntegrationAsync(unittest.IsolatedAsyncioTestCase):
    """Async integration test suite for PromptsClient."""

    @classmethod
    def setUpClass(cls):
        """Set up integration test fixtures."""
        cls.api_key = os.getenv("BASALT_API_KEY")
        if not cls.api_key:
            raise unittest.SkipTest("BASALT_API_KEY environment variable not set")

        cls.test_prompt_slug = os.getenv("BASALT_TEST_PROMPT_SLUG")
        if not cls.test_prompt_slug:
            raise unittest.SkipTest("BASALT_TEST_PROMPT_SLUG environment variable not set")

        cls.logger = Logger(log_level="all")
        cls.cache = MemoryCache()
        cls.fallback_cache = MemoryCache()

        cls.client = PromptsClient(
            api_key=cls.api_key,
            cache=cls.cache,
            fallback_cache=cls.fallback_cache,
            logger=cls.logger,
        )

    async def test_get_async_real_api(self):
        """Test asynchronous prompt retrieval with real API."""
        prompt, generation = await self.client.get(self.test_prompt_slug)

        # Verify prompt object
        self.assertIsInstance(prompt, Prompt)
        self.assertEqual(prompt.slug, self.test_prompt_slug)
        self.assertIsNotNone(prompt.text)
        self.assertIsNotNone(prompt.model)

    async def test_describe_async_real_api(self):
        """Test async describe with real API."""
        response = await self.client.describe(self.test_prompt_slug)

        self.assertIsInstance(response, DescribePromptResponse)
        self.assertEqual(response.slug, self.test_prompt_slug)

    async def test_list_async_real_api(self):
        """Test async list with real API."""
        prompts = await self.client.list()

        self.assertIsInstance(prompts, list)
        if prompts:
            self.assertIsInstance(prompts[0], PromptListResponse)

    async def test_get_async_with_variables_real_api(self):
        """Test async prompt retrieval with variables."""
        describe_response = await self.client.describe(self.test_prompt_slug)

        if describe_response.variables:
            variables = {
                var["name"]: f"test_{var['name']}"
                for var in describe_response.variables[:2]
            }

            prompt, generation = await self.client.get(
                self.test_prompt_slug,
                variables=variables
            )

            self.assertIsNotNone(prompt.variables)


if __name__ == "__main__":
    unittest.main()
