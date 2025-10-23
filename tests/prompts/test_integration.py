"""
Integration tests for PromptsClient with real API.

These tests make real API requests to the Basalt backend. They can be toggled
using the BASALT_RUN_INTEGRATION_TESTS environment variable.

To run integration tests:
    export BASALT_RUN_INTEGRATION_TESTS=1
    export BASALT_API_KEY=your-api-key
    export BASALT_TEST_PROMPT_SLUG=your-test-prompt-slug
    python -m pytest tests/prompts/test_integration.py
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

from basalt._internal.exceptions import NotFoundError
from basalt.prompts.client import PromptsClient
from basalt.prompts.models import DescribePromptResponse, Prompt, PromptListResponse, PublishPromptResponse
from basalt.utils.memcache import MemoryCache


# Typed container for fixture return
@dataclass
class PromptIntegrationContext:
    """Typed context for prompt integration tests."""

    api_key: str
    test_prompt_slug: str
    cache: MemoryCache
    fallback_cache: MemoryCache
    client: PromptsClient


@pytest.fixture(scope="class")
def prompt_ctx() -> PromptIntegrationContext:
    """Set up integration test fixtures and return a typed context."""
    api_key = os.getenv("BASALT_API_KEY")
    if not api_key:
        pytest.skip("BASALT_API_KEY not set")

    test_prompt_slug = os.getenv("BASALT_TEST_PROMPT_SLUG")
    if not test_prompt_slug:
        pytest.skip("BASALT_TEST_PROMPT_SLUG not set")

    cache = MemoryCache()
    fallback_cache = MemoryCache()
    client = PromptsClient(
        api_key=api_key,
        cache=cache,
        fallback_cache=fallback_cache,
    )

    return PromptIntegrationContext(
        api_key=api_key,
        test_prompt_slug=test_prompt_slug,
        cache=cache,
        fallback_cache=fallback_cache,
        client=client,
    )


@pytest.mark.skipif(
    os.getenv("BASALT_RUN_INTEGRATION_TESTS") != "1",
    reason="Integration tests disabled. Set BASALT_RUN_INTEGRATION_TESTS=1"
)
class TestPromptsClientIntegration:
    """Integration test suite for PromptsClient."""

    def test_get_sync_real_api(self, prompt_ctx: PromptIntegrationContext) -> None:
        """Test synchronous prompt retrieval with real API."""
        prompt = prompt_ctx.client.get_sync(prompt_ctx.test_prompt_slug)

        # Verify prompt object
        assert isinstance(prompt, Prompt)
        assert prompt.slug == prompt_ctx.test_prompt_slug
        assert prompt.text is not None
        assert prompt.model is not None
        assert prompt.version is not None


    def test_get_sync_with_version_real_api(self, prompt_ctx: PromptIntegrationContext) -> None:
        """Test prompt retrieval with specific version."""
        describe_response = prompt_ctx.client.describe_sync(prompt_ctx.test_prompt_slug)

        if describe_response.available_versions:
            version = describe_response.available_versions[0]
            prompt= prompt_ctx.client.get_sync(
                prompt_ctx.test_prompt_slug,
                version=version,
            )

            assert prompt.version == version

    def test_get_sync_with_tag_real_api(self, prompt_ctx: PromptIntegrationContext) -> None:
        """Test prompt retrieval with tag."""
        describe_response = prompt_ctx.client.describe_sync(prompt_ctx.test_prompt_slug)

        if describe_response.available_tags:
            tag = describe_response.available_tags[0]
            prompt = prompt_ctx.client.get_sync(
                prompt_ctx.test_prompt_slug,
                tag=tag,
            )

            assert prompt.tag == tag

    def test_get_sync_with_variables_real_api(self, prompt_ctx: PromptIntegrationContext) -> None:
        """Test prompt retrieval with variable substitution."""
        describe_response = prompt_ctx.client.describe_sync(prompt_ctx.test_prompt_slug)

        if describe_response.variables:
            # Create variables dict from available variables
            variables: dict[str, str] = {
                var["name"]: f"test_{var['name']}"
                for var in describe_response.variables[:3]
            }

            prompt = prompt_ctx.client.get_sync(
                prompt_ctx.test_prompt_slug,
                variables=variables,
            )

            # Verify variables were applied
            assert prompt.variables is not None
            for _key, value in variables.items():
                assert value in prompt.text

    def test_get_sync_cache_works_real_api(self, prompt_ctx: PromptIntegrationContext) -> None:
        """Test that caching works with real API."""
        # Clear cache first
        cache = MemoryCache()
        prompt_ctx.client._cache = cache

        # First request - should hit API
        prompt1 = prompt_ctx.client.get_sync(prompt_ctx.test_prompt_slug)

        # Verify cache was populated
        cache_key = (prompt_ctx.test_prompt_slug, None, None)
        cached_value = cache.get(cache_key)
        assert cached_value is not None

        # Second request - should use cache
        prompt2 = prompt_ctx.client.get_sync(prompt_ctx.test_prompt_slug)

        # Verify both prompts are the same
        assert prompt1.slug == prompt2.slug
        assert prompt1.text == prompt2.text

    def test_get_sync_not_found_real_api(self, prompt_ctx: PromptIntegrationContext) -> None:
        """Test 404 error handling with real API."""
        with pytest.raises(NotFoundError):
            prompt_ctx.client.get_sync("nonexistent-prompt-slug-12345")

    def test_describe_sync_real_api(self, prompt_ctx: PromptIntegrationContext) -> None:
        """Test describe method with real API."""
        response = prompt_ctx.client.describe_sync(prompt_ctx.test_prompt_slug)

        # Verify response
        assert isinstance(response, DescribePromptResponse)
        assert response.slug == prompt_ctx.test_prompt_slug
        assert response.name is not None
        assert response.status is not None
        assert isinstance(response.available_versions, list)
        assert isinstance(response.available_tags, list)
        assert isinstance(response.variables, list)

    def test_list_sync_real_api(self, prompt_ctx: PromptIntegrationContext) -> None:
        """Test list method with real API."""
        prompts = prompt_ctx.client.list_sync()

        # Verify response
        assert isinstance(prompts, list)
        if prompts:  # If there are prompts
            assert isinstance(prompts[0], PromptListResponse)
            assert prompts[0].slug is not None
            assert prompts[0].name is not None
            assert prompts[0].status is not None

    def test_fallback_cache_real_api(self, prompt_ctx: PromptIntegrationContext) -> None:
        """Test fallback cache with real API."""
        # First, populate the fallback cache with a successful request
        prompt1 = prompt_ctx.client.get_sync(prompt_ctx.test_prompt_slug)

        # Now use an invalid API key to force an error
        bad_client = PromptsClient(
            api_key="invalid-key",
            cache=MemoryCache(),  # Empty primary cache
            fallback_cache=prompt_ctx.fallback_cache,  # Use populated fallback cache
        )

        # This should use the fallback cache instead of failing
        prompt2 = bad_client.get_sync(prompt_ctx.test_prompt_slug)

        # Verify we got the cached prompt
        assert prompt1.slug == prompt2.slug
        assert prompt1.text == prompt2.text

    def test_cache_disabled_real_api(self, prompt_ctx: PromptIntegrationContext) -> None:
        """Test that cache can be disabled."""
        # Populate cache
        prompt_ctx.client.get_sync(prompt_ctx.test_prompt_slug)

        # Clear cache counters
        cache = MemoryCache()
        prompt_ctx.client._cache = cache

        # Request with cache disabled should still work
        prompt_ctx.client.get_sync(prompt_ctx.test_prompt_slug, cache_enabled=False)

        # Verify cache was not used
        cache_key = (prompt_ctx.test_prompt_slug, None, None)
        assert cache.get(cache_key) is None

    def test_publish_prompt_sync_real_api(self, prompt_ctx: PromptIntegrationContext) -> None:
        """Test synchronous prompt publishing with real API."""
        # Get available versions first
        describe_response = prompt_ctx.client.describe_sync(prompt_ctx.test_prompt_slug)

        if not describe_response.available_versions:
            pytest.skip("No versions available for test prompt")

        version = describe_response.available_versions[0]

        # Generate a unique tag name for testing
        import time
        test_tag = f"test-tag-{int(time.time())}"

        # Publish the prompt
        response = prompt_ctx.client.publish_prompt_sync(
            slug=prompt_ctx.test_prompt_slug,
            new_tag=test_tag,
            version=version,
        )

        # Verify response
        assert isinstance(response, PublishPromptResponse)
        assert response.id is not None
        assert response.label == test_tag

    def test_publish_prompt_sync_with_tag_real_api(self, prompt_ctx: PromptIntegrationContext) -> None:
        """Test publishing from an existing tag."""
        describe_response = prompt_ctx.client.describe_sync(prompt_ctx.test_prompt_slug)

        if not describe_response.available_tags:
            pytest.skip("No tags available for test prompt")

        tag = describe_response.available_tags[0]

        # Generate a unique tag name
        import time
        new_tag = f"test-from-tag-{int(time.time())}"

        # Publish from existing tag
        response = prompt_ctx.client.publish_prompt_sync(
            slug=prompt_ctx.test_prompt_slug,
            new_tag=new_tag,
            tag=tag,
        )

        assert isinstance(response, PublishPromptResponse)
        assert response.label == new_tag


@pytest.mark.skipif(
    os.getenv("BASALT_RUN_INTEGRATION_TESTS") != "1",
    reason="Integration tests disabled. Set BASALT_RUN_INTEGRATION_TESTS=1"
)
class TestPromptsClientIntegrationAsync:
    """Async integration test suite for PromptsClient."""

    @pytest.mark.asyncio
    async def test_get_async_real_api(self, prompt_ctx: PromptIntegrationContext) -> None:
        """Test asynchronous prompt retrieval with real API."""
        prompt = await prompt_ctx.client.get(prompt_ctx.test_prompt_slug)

        # Verify prompt object
        assert isinstance(prompt, Prompt)
        assert prompt.slug == prompt_ctx.test_prompt_slug
        assert prompt.text is not None
        assert prompt.model is not None

    @pytest.mark.asyncio
    async def test_describe_async_real_api(self, prompt_ctx: PromptIntegrationContext) -> None:
        """Test async describe with real API."""
        response = await prompt_ctx.client.describe(prompt_ctx.test_prompt_slug)

        assert isinstance(response, DescribePromptResponse)
        assert response.slug == prompt_ctx.test_prompt_slug

    @pytest.mark.asyncio
    async def test_list_async_real_api(self, prompt_ctx: PromptIntegrationContext) -> None:
        """Test async list with real API."""
        prompts = await prompt_ctx.client.list()

        assert isinstance(prompts, list)
        if prompts:
            assert isinstance(prompts[0], PromptListResponse)

    @pytest.mark.asyncio
    async def test_get_async_with_variables_real_api(self, prompt_ctx: PromptIntegrationContext) -> None:
        """Test async prompt retrieval with variables."""
        describe_response = await prompt_ctx.client.describe(prompt_ctx.test_prompt_slug)

        if describe_response.variables:
            variables: dict[str, str] = {
                var["name"]: f"test_{var['name']}"
                for var in describe_response.variables[:2]
            }

            prompt = await prompt_ctx.client.get(
                prompt_ctx.test_prompt_slug,
                variables=variables,
            )

            assert prompt.variables is not None

    @pytest.mark.asyncio
    async def test_publish_prompt_async_real_api(self, prompt_ctx: PromptIntegrationContext) -> None:
        """Test asynchronous prompt publishing with real API."""
        describe_response = await prompt_ctx.client.describe(prompt_ctx.test_prompt_slug)

        if not describe_response.available_versions:
            pytest.skip("No versions available for test prompt")

        version = describe_response.available_versions[0]

        # Generate a unique tag name
        import time
        test_tag = f"test-async-{int(time.time())}"

        # Publish the prompt
        response = await prompt_ctx.client.publish_prompt(
            slug=prompt_ctx.test_prompt_slug,
            new_tag=test_tag,
            version=version,
        )

        # Verify response
        assert isinstance(response, PublishPromptResponse)
        assert response.id is not None
        assert response.label == test_tag

    @pytest.mark.asyncio
    async def test_publish_prompt_async_with_both_version_and_tag_real_api(
        self, prompt_ctx: PromptIntegrationContext
    ) -> None:
        """Test publishing with both version and tag specified."""
        describe_response = await prompt_ctx.client.describe(prompt_ctx.test_prompt_slug)

        if not describe_response.available_versions:
            pytest.skip("No versions available for test prompt")
        if not describe_response.available_tags:
            pytest.skip("No tags available for test prompt")

        version = describe_response.available_versions[0]
        tag = describe_response.available_tags[0]

        # Generate a unique tag name
        import time
        new_tag = f"test-both-{int(time.time())}"

        # Publish with both version and tag
        response = await prompt_ctx.client.publish_prompt(
            slug=prompt_ctx.test_prompt_slug,
            new_tag=new_tag,
            version=version,
            tag=tag,
        )

        assert isinstance(response, PublishPromptResponse)
        assert response.label == new_tag
