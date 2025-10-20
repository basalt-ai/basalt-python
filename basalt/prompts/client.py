"""
Prompts API Client.

This module provides the PromptsClient for interacting with the Basalt Prompts API.
"""
from __future__ import annotations

from typing import cast

from .._internal.exceptions import BasaltAPIError
from .._internal.http import HTTPClient
from ..config import config
from ..objects.prompt import Prompt as PromptObject
from ..resources.prompts import prompt_types as resource_prompt_types
from ..resources.prompts.prompt_types import PromptParams
from ..utils.protocols import ICache
from .models import (
    DescribePromptResponse,
    Prompt,
    PromptListResponse,
    PromptResponse,
)


class PromptsClient:
    """
    Client for interacting with the Basalt Prompts API.

    This client provides methods to retrieve, describe, and list prompts with
    caching support and monitoring integration.
    """

    def __init__(
        self,
        api_key: str,
        cache: ICache,
        fallback_cache: ICache,
        base_url: str | None = None,
    ):
        """
        Initialize the PromptsClient.

        Args:
            api_key: The Basalt API key for authentication.
            cache: Primary cache instance for storing prompt responses.
            fallback_cache: Fallback cache for graceful degradation on API failures.
            base_url: Optional base URL for the API (defaults to config value).
        """
        self._api_key = api_key
        self._cache = cache
        self._fallback_cache = fallback_cache
        self._base_url = base_url or config["api_url"]
        self._http_client = HTTPClient()

        # Cache responses for 5 minutes
        self._cache_duration = 5 * 60

    async def get(
        self,
        slug: str,
        version: str | None = None,
        tag: str | None = None,
        variables: dict[str, str] | None = None,
        cache_enabled: bool = True,
    ) -> Prompt:
        """
        Retrieve a prompt by slug, optionally specifying version and tag.

        Args:
            slug: The slug identifier for the prompt.
            version: The version of the prompt (optional).
            tag: The tag associated with the prompt (optional).
            variables: A dictionary of variables to replace in the prompt text (optional).
            cache_enabled: Enable or disable cache for this request (default: True).

        Returns:
            A tuple containing the Prompt and Generation object.

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
                return prompt

        # Make API request
        try:
            url = f"{self._base_url}/prompts/{slug}"
            params = {}
            if version:
                params["version"] = version
            if tag:
                params["tag"] = tag

            response = await HTTPClient.fetch(
                url=url,
                method="GET",
                params=params,
                headers=self._get_headers(),
            )

            prompt_data = response.get("prompt", {})
            prompt_response = PromptResponse.from_dict(prompt_data)

            # Store in both caches
            if cache_enabled:
                self._cache.put(cache_key, prompt_response, self._cache_duration)
                # Also store in fallback cache with the same duration so the
                # fallback can be used for the same TTL when API errors occur.
                self._fallback_cache.put(cache_key, prompt_response, self._cache_duration)

            prompt = self._create_prompt_instance(prompt_response, variables)
            return prompt

        except BasaltAPIError as e:
            # Try fallback cache
            if cache_enabled:
                fallback = self._fallback_cache.get(cache_key)
                if fallback:
                    prompt_response = cast(PromptResponse, fallback)
                    prompt = self._create_prompt_instance(prompt_response, variables)
                    return prompt
            raise BasaltAPIError("Failed to retrieve prompt") from e

    def get_sync(
        self,
        slug: str,
        version: str | None = None,
        tag: str | None = None,
        variables: dict[str, str] | None = None,
        cache_enabled: bool = True,
    ) -> Prompt:
        """
        Synchronously retrieve a prompt by slug, optionally specifying version and tag.

        Args:
            slug: The slug identifier for the prompt.
            version: The version of the prompt (optional).
            tag: The tag associated with the prompt (optional).
            variables: A dictionary of variables to replace in the prompt text (optional).
            cache_enabled: Enable or disable cache for this request (default: True).

        Returns:
            A tuple containing the Prompt and Generation object.

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
                return prompt

        # Make API request
        try:
            url = f"{self._base_url}/prompts/{slug}"
            params = {}
            if version:
                params["version"] = version
            if tag:
                params["tag"] = tag

            response = HTTPClient.fetch_sync(
                url=url,
                method="GET",
                params=params,
                headers=self._get_headers(),
            )

            prompt_data = response.get("prompt", {})
            prompt_response = PromptResponse.from_dict(prompt_data)

            # Store in both caches
            if cache_enabled:
                self._cache.put(cache_key, prompt_response, self._cache_duration)
                # Mirror the TTL to fallback cache as well.
                self._fallback_cache.put(cache_key, prompt_response, self._cache_duration)

            prompt = self._create_prompt_instance(prompt_response, variables)
            return prompt

        except BasaltAPIError as e:
            # Try fallback cache
            if cache_enabled:
                fallback = self._fallback_cache.get(cache_key)
                if fallback:
                    prompt_response = cast(PromptResponse, fallback)
                    prompt = self._create_prompt_instance(prompt_response, variables)
                    return prompt

            raise BasaltAPIError("Failed to retrieve prompt") from e

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

        response = await HTTPClient.fetch(
            url=url,
            method="GET",
            params=params,
            headers=self._get_headers(),
        )

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

        response = HTTPClient.fetch_sync(
            url=url,
            method="GET",
            params=params,
            headers=self._get_headers(),
        )

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

        response = await HTTPClient.fetch(
            url=url,
            method="GET",
            params=params,
            headers=self._get_headers(),
        )

        prompts_data = response.get("prompts", [])
        return [PromptListResponse.from_dict(p) for p in prompts_data]

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

        response = HTTPClient.fetch_sync(
            url=url,
            method="GET",
            params=params,
            headers=self._get_headers(),
        )

        prompts_data = response.get("prompts", [])
        return [PromptListResponse.from_dict(p) for p in prompts_data]


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
        # Convert internal PromptModel to resource PromptModel expected by PromptParams
        prm = prompt_response.model
        resource_model = resource_prompt_types.PromptModel(
            provider=prm.provider,
            model=prm.model,
            version=prm.version,
            parameters=resource_prompt_types.PromptModelParameters(
                temperature=prm.parameters.temperature,
                max_length=prm.parameters.max_length,
                response_format=prm.parameters.response_format,
                top_k=prm.parameters.top_k,
                top_p=prm.parameters.top_p,
                frequency_penalty=prm.parameters.frequency_penalty,
                presence_penalty=prm.parameters.presence_penalty,
                json_object=prm.parameters.json_object,
            ),
        )

        prompt_obj = PromptObject(PromptParams(
            slug=prompt_response.slug,
            text=prompt_response.text,
            tag=prompt_response.tag,
            model=resource_model,
            version=prompt_response.version,
            system_text=prompt_response.system_text,
            variables=variables,
        ))

        # Return the dataclass Prompt, not the PromptObject
        return Prompt(
            slug=prompt_obj.slug,
            text=prompt_obj.text,
            raw_text=prompt_obj.raw_text,
            model=prompt_response.model,
            version=prompt_obj.version,
            system_text=prompt_obj.system_text,
            raw_system_text=prompt_obj.raw_system_text,
            variables=prompt_obj.variables,
            tag=prompt_obj.tag,
        )

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
