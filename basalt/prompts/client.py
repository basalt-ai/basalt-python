"""
Prompts API Client.

This module provides the PromptsClient for interacting with the Basalt Prompts API.
"""
from __future__ import annotations

from typing import Any, cast

from .._internal.base_client import BaseServiceClient
from .._internal.http import HTTPClient
from ..config import config
from ..observability.spans import BasaltRequestSpan
from ..types.cache import CacheProtocol
from ..types.exceptions import BasaltAPIError
from .models import (
    AsyncPromptContextManager,
    DescribePromptResponse,
    Prompt,
    PromptContextManager,
    PromptListResponse,
    PromptResponse,
    PublishPromptResponse,
)


class PromptRequestSpan(BasaltRequestSpan):
    """Span metadata for prompt API requests with JSON output."""

    def format_output(self, response_data: dict[str, Any]) -> dict[str, Any]:
        """Return the Prompt object as JSON output."""
        prompt_data = response_data.get("prompt", {})
        if not prompt_data:
            # Fallback to status code if no prompt data
            status_code = getattr(response_data, "status_code", None)
            return {"status_code": status_code}

        # Return the full prompt object as JSON
        return {
            "prompt": prompt_data,
            "from_cache": response_data.get("from_cache", False)
        }


class PromptsClient(BaseServiceClient):
    """
    Client for interacting with the Basalt Prompts API.

    This client provides methods to retrieve, describe, and list prompts with
    caching support and monitoring integration.
    """

    def __init__(
        self,
        api_key: str,
        cache: CacheProtocol,
        fallback_cache: CacheProtocol,
        base_url: str | None = None,
        http_client: HTTPClient | None = None,
        log_level: str | None = None,
    ):
        """
        Initialize the PromptsClient.

        Args:
            api_key: The Basalt API key for authentication.
            cache: Primary cache instance for storing prompt responses.
            fallback_cache: Fallback cache for graceful degradation on API failures.
            base_url: Optional base URL for the API (defaults to config value).
            log_level: Optional log level for the client logger.
        """
        self._api_key = api_key
        self._cache = cache
        self._fallback_cache = fallback_cache
        self._base_url = base_url or config["api_url"]
        super().__init__(client_name="prompts", http_client=http_client, log_level=log_level)

        # Cache responses for 5 minutes
        self._cache_duration = 5 * 60

    async def _request_async(
        self,
        operation: str,
        *,
        method: str,
        url: str,
        span_attributes: dict[str, Any] | None = None,
        span_variables: dict[str, Any] | None = None,
        cache_hit: bool | None = None,
        **request_kwargs: Any,
    ):
        """Override to use PromptRequestSpan for custom output formatting."""
        # Lazy import to avoid circular dependency
        import functools

        from basalt.observability.request_tracing import trace_async_request

        span = PromptRequestSpan(
            client=self._client_name,
            operation=operation,
            method=method,
            url=url,
            cache_hit=cache_hit,
            extra_attributes=self._filter_attributes(span_attributes),
            variables=self._filter_attributes(span_variables),
        )
        call = functools.partial(
            self._http_client.fetch,
            url=url,
            method=method,
            **request_kwargs,
        )
        return await trace_async_request(span, call)

    def _request_sync(
        self,
        operation: str,
        *,
        method: str,
        url: str,
        span_attributes: dict[str, Any] | None = None,
        span_variables: dict[str, Any] | None = None,
        cache_hit: bool | None = None,
        **request_kwargs: Any,
    ):
        """Override to use PromptRequestSpan for custom output formatting."""
        # Lazy import to avoid circular dependency
        import functools

        from basalt.observability.request_tracing import trace_sync_request

        span = PromptRequestSpan(
            client=self._client_name,
            operation=operation,
            method=method,
            url=url,
            cache_hit=cache_hit,
            extra_attributes=self._filter_attributes(span_attributes),
            variables=self._filter_attributes(span_variables),
        )
        call = functools.partial(
            self._http_client.fetch_sync,
            url=url,
            method=method,
            **request_kwargs,
        )
        return trace_sync_request(span, call)

    async def get(
        self,
        slug: str,
        version: str | None = None,
        tag: str | None = None,
        variables: dict[str, str] | None = None,
        cache_enabled: bool = True,
    ) -> AsyncPromptContextManager:
        """
        Retrieve a prompt by slug, optionally specifying version and tag.

        Args:
            slug: The slug identifier for the prompt.
            version: The version of the prompt (optional).
            tag: The tag associated with the prompt (optional).
            variables: A dictionary of variables to replace in the prompt text (optional).
            cache_enabled: Enable or disable cache for this request (default: True).

        Returns:
            An AsyncPromptContextManager wrapping the Prompt. Can be used directly or as an async context manager.

        Raises:
            BasaltAPIError: If the API request fails and no fallback cache is available.
            NetworkError: If a network error occurs.
        """
        cache_key = (slug, version, tag)

        # Check primary cache
        if cache_enabled:
            cached = self._cache.get(cache_key)
            if cached:
                prompt_response = cast(PromptResponse, cached)
                prompt = self._create_prompt_instance(prompt_response, variables)
                return AsyncPromptContextManager(
                    prompt=prompt,
                    slug=slug,
                    version=version,
                    tag=tag,
                    variables=variables,
                    from_cache=True,
                )

        # Make API request
        try:
            url = f"{self._base_url}/prompts/{slug}"
            params = {}
            if version:
                params["version"] = version
            if tag:
                params["tag"] = tag

            response = await self._request_async(
                "get",
                method="GET",
                url=url,
                params=params,
                headers=self._get_headers(),
                span_attributes={
                    "basalt.prompt.slug": slug,
                    "basalt.prompt.version": version,
                    "basalt.prompt.tag": tag,
                },
                span_variables=variables,
            )

            if response is None or response.body is None:
                raise BasaltAPIError("Empty response from get prompt API")
            prompt_data = response.get("prompt", {})
            prompt_response = PromptResponse.from_dict(prompt_data)

            # Store in both caches
            if cache_enabled:
                self._cache.put(cache_key, prompt_response, self._cache_duration)
                # Also store in fallback cache with the same duration so the
                # fallback can be used for the same TTL when API errors occur.
                self._fallback_cache.put(cache_key, prompt_response, self._cache_duration)

            prompt = self._create_prompt_instance(prompt_response, variables)
            return AsyncPromptContextManager(
                prompt=prompt,
                slug=slug,
                version=version,
                tag=tag,
                variables=variables,
                from_cache=False,
            )

        except BasaltAPIError:
            # Try fallback cache
            if cache_enabled:
                fallback = self._fallback_cache.get(cache_key)
                if fallback:
                    prompt_response = cast(PromptResponse, fallback)
                    prompt = self._create_prompt_instance(prompt_response, variables)
                    return AsyncPromptContextManager(
                        prompt=prompt,
                        slug=slug,
                        version=version,
                        tag=tag,
                        variables=variables,
                        from_cache=True,
                    )
            raise  # Re-raise the original API error

    def get_sync(
        self,
        slug: str,
        version: str | None = None,
        tag: str | None = None,
        variables: dict[str, str] | None = None,
        cache_enabled: bool = True,
    ) -> PromptContextManager:
        """
        Synchronously retrieve a prompt by slug, optionally specifying version and tag.

        Args:
            slug: The slug identifier for the prompt.
            version: The version of the prompt (optional).
            tag: The tag associated with the prompt (optional).
            variables: A dictionary of variables to replace in the prompt text (optional).
            cache_enabled: Enable or disable cache for this request (default: True).

        Returns:
            A PromptContextManager wrapping the Prompt. Can be used directly or as a context manager.

        Raises:
            BasaltAPIError: If the API request fails and no fallback cache is available.
            NetworkError: If a network error occurs.
        """
        cache_key = (slug, version, tag)

        # Check primary cache
        if cache_enabled:
            cached = self._cache.get(cache_key)
            if cached:
                prompt_response = cast(PromptResponse, cached)
                prompt = self._create_prompt_instance(prompt_response, variables)
                return PromptContextManager(
                    prompt=prompt,
                    slug=slug,
                    version=version,
                    tag=tag,
                    variables=variables,
                    from_cache=True,
                )

        # Make API request
        try:
            url = f"{self._base_url}/prompts/{slug}"
            params = {}
            if version:
                params["version"] = version
            if tag:
                params["tag"] = tag

            response = self._request_sync(
                "get",
                method="GET",
                url=url,
                params=params,
                headers=self._get_headers(),
                span_attributes={
                    "basalt.prompt.slug": slug,
                    "basalt.prompt.version": version,
                    "basalt.prompt.tag": tag,
                },
                span_variables=variables,
            )

            if response is None or response.body is None:
                raise BasaltAPIError("Empty response from get prompt API")
            prompt_data = response.get("prompt", {})
            prompt_response = PromptResponse.from_dict(prompt_data)

            # Store in both caches
            if cache_enabled:
                self._cache.put(cache_key, prompt_response, self._cache_duration)
                # Mirror the TTL to fallback cache as well.
                self._fallback_cache.put(cache_key, prompt_response, self._cache_duration)

            prompt = self._create_prompt_instance(prompt_response, variables)
            return PromptContextManager(
                prompt=prompt,
                slug=slug,
                version=version,
                tag=tag,
                variables=variables,
                from_cache=False,
            )

        except BasaltAPIError:
            # Try fallback cache
            if cache_enabled:
                fallback = self._fallback_cache.get(cache_key)
                if fallback:
                    prompt_response = cast(PromptResponse, fallback)
                    prompt = self._create_prompt_instance(prompt_response, variables)
                    return PromptContextManager(
                        prompt=prompt,
                        slug=slug,
                        version=version,
                        tag=tag,
                        variables=variables,
                        from_cache=True,
                    )

            raise  # Re-raise the original API error

    async def describe(
        self,
        slug: str,
        version: str | None = None,
        tag: str | None = None,
    ) -> DescribePromptResponse:
        """
        Get details about a prompt by slug, optionally specifying version and tag.

        Args:
            slug: The slug identifier for the prompt.
            version: The version of the prompt (optional).
            tag: The tag associated with the prompt (optional).

        Returns:
            DescribePromptResponse containing prompt metadata.

        Raises:
            BasaltAPIError: If the API request fails.
            NetworkError: If a network error occurs.
        """
        url = f"{self._base_url}/prompts/{slug}/describe"
        params = {}
        if version:
            params["version"] = version
        if tag:
            params["tag"] = tag

        response = await self._request_async(
            "describe",
            method="GET",
            url=url,
            params=params,
            headers=self._get_headers(),
            span_attributes={
                "basalt.prompt.slug": slug,
                "basalt.prompt.version": version,
                "basalt.prompt.tag": tag,
            },
        )

        if response is None or response.body is None:
            raise BasaltAPIError("Empty response from describe prompt API")
        prompt_data = response.get("prompt", {})
        return DescribePromptResponse.from_dict(prompt_data)

    def describe_sync(
        self,
        slug: str,
        version: str | None = None,
        tag: str | None = None,
    ) -> DescribePromptResponse:
        """
        Synchronously get details about a prompt by slug, optionally specifying version and tag.

        Args:
            slug: The slug identifier for the prompt.
            version: The version of the prompt (optional).
            tag: The tag associated with the prompt (optional).

        Returns:
            DescribePromptResponse containing prompt metadata.

        Raises:
            BasaltAPIError: If the API request fails.
            NetworkError: If a network error occurs.
        """
        url = f"{self._base_url}/prompts/{slug}/describe"
        params = {}
        if version:
            params["version"] = version
        if tag:
            params["tag"] = tag

        response = self._request_sync(
            "describe",
            method="GET",
            url=url,
            params=params,
            headers=self._get_headers(),
            span_attributes={
                "basalt.prompt.slug": slug,
                "basalt.prompt.version": version,
                "basalt.prompt.tag": tag,
            },
        )

        if response is None or response.body is None:
            raise BasaltAPIError("Empty response from describe prompt API")
        prompt_data = response.get("prompt", {})
        return DescribePromptResponse.from_dict(prompt_data)

    async def list(self, feature_slug: str | None = None) -> list[PromptListResponse]:
        """
        List prompts, optionally filtering by feature_slug.

        Args:
            feature_slug: Optional feature slug to filter prompts by.

        Returns:
            A list of PromptListResponse objects.

        Raises:
            BasaltAPIError: If the API request fails.
            NetworkError: If a network error occurs.
        """
        url = f"{self._base_url}/prompts"
        params = {}
        if feature_slug:
            params["featureSlug"] = feature_slug

        response = await self._request_async(
            "list",
            method="GET",
            url=url,
            params=params,
            headers=self._get_headers(),
            span_attributes={
                "basalt.prompt.feature_slug": feature_slug,
            },
        )

        if response is None or response.body is None:
            return []

        prompts_data = response.get("prompts", [])
        return [
            PromptListResponse.from_dict(p)
            for p in prompts_data
            if isinstance(p, dict)
        ]

    def list_sync(self, feature_slug: str | None = None) -> list[PromptListResponse]:
        """
        Synchronously list prompts, optionally filtering by feature_slug.

        Args:
            feature_slug: Optional feature slug to filter prompts by.

        Returns:
            A list of PromptListResponse objects.

        Raises:
            BasaltAPIError: If the API request fails.
            NetworkError: If a network error occurs.
        """
        url = f"{self._base_url}/prompts"
        params = {}
        if feature_slug:
            params["featureSlug"] = feature_slug

        response = self._request_sync(
            "list",
            method="GET",
            url=url,
            params=params,
            headers=self._get_headers(),
            span_attributes={
                "basalt.prompt.feature_slug": feature_slug,
            },
        )

        if response is None or response.body is None:
            return []

        prompts_data = response.get("prompts", [])
        return [
            PromptListResponse.from_dict(p)
            for p in prompts_data
            if isinstance(p, dict)
        ]

    async def publish(
        self,
        slug: str,
        new_tag: str,
        version: str | None = None,
        tag: str | None = None,
    ) -> PublishPromptResponse:
        """
        Publish a prompt with a new deployment tag.

        Args:
            slug: The slug identifier for the prompt.
            new_tag: The new deployment tag to create.
            version: The version of the prompt to publish (optional).
            tag: The existing tag to publish from (optional).

        Returns:
            PublishPromptResponse containing the deployment tag information.

        Raises:
            BasaltAPIError: If the API request fails.
            NetworkError: If a network error occurs.
        """
        url = f"{self._base_url}/prompts/{slug}/publish"
        body = {"newTag": new_tag}

        if version:
            body["version"] = version
        if tag:
            body["tag"] = tag

        response = await self._request_async(
            "publish_prompt",
            method="POST",
            url=url,
            body=body,
            headers=self._get_headers(),
            span_attributes={
                "basalt.prompt.slug": slug,
                "basalt.prompt.new_tag": new_tag,
                "basalt.prompt.version": version,
                "basalt.prompt.tag": tag,
            },
        )

        if response is None:
            raise BasaltAPIError("Empty response from publish prompt API")
        payload = response.json() or {}
        if not payload:
            raise BasaltAPIError("Empty response from publish prompt API")

        return PublishPromptResponse.from_dict(payload)

    def publish_sync(
        self,
        slug: str,
        new_tag: str,
        version: str | None = None,
        tag: str | None = None,
    ) -> PublishPromptResponse:
        """
        Synchronously publish a prompt with a new deployment tag.

        Args:
            slug: The slug identifier for the prompt.
            new_tag: The new deployment tag to create.
            version: The version of the prompt to publish (optional).
            tag: The existing tag to publish from (optional).

        Returns:
            PublishPromptResponse containing the deployment tag information.

        Raises:
            BasaltAPIError: If the API request fails.
            NetworkError: If a network error occurs.
        """
        url = f"{self._base_url}/prompts/{slug}/publish"
        body = {"newTag": new_tag}

        if version:
            body["version"] = version
        if tag:
            body["tag"] = tag

        response = self._request_sync(
            "publish_prompt",
            method="POST",
            url=url,
            body=body,
            headers=self._get_headers(),
            span_attributes={
                "basalt.prompt.slug": slug,
                "basalt.prompt.new_tag": new_tag,
                "basalt.prompt.version": version,
                "basalt.prompt.tag": tag,
            },
        )

        if response is None:
            raise BasaltAPIError("Empty response from publish prompt API")
        payload = response.json() or {}
        if not payload:
            raise BasaltAPIError("Empty response from publish prompt API")

        return PublishPromptResponse.from_dict(payload)

    @staticmethod
    def _create_prompt_instance(
        prompt_response: PromptResponse,
        variables: dict | None = None,
    ) -> Prompt:
        """
        Create a Prompt instance from a PromptResponse.

        Args:
            prompt_response: The API response containing prompt data.
            variables: Optional variables to compile into the prompt.

        Returns:
            Prompt instance with compiled variables if provided.
        """
        # Create the Prompt dataclass instance
        prompt = Prompt(
            slug=prompt_response.slug,
            text=prompt_response.text,
            raw_text=prompt_response.text,
            model=prompt_response.model,
            version=prompt_response.version,
            system_text=prompt_response.system_text,
            raw_system_text=prompt_response.system_text,
            variables=variables,
            tag=prompt_response.tag,
        )

        # Compile variables if provided
        if variables:
            prompt.compile_variables(variables)

        return prompt

    def _get_headers(self) -> dict[str, str]:
        """
        Get the HTTP headers for API requests.

        Returns:
            Dictionary of HTTP headers.
        """
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-BASALT-SDK-VERSION": config["sdk_version"],
            "X-BASALT-SDK-TYPE": config["sdk_type"],
        }
