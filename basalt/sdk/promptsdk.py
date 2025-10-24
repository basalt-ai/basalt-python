from typing import Optional, Dict, Tuple, cast

from ..ressources.monitor.generation_types import GenerationParams, PromptReference
from ..ressources.monitor.trace_types import TraceParams
from ..ressources.prompts.prompt_types import Prompt as IPrompt, PromptParams
from ..utils.dtos import (
	GetPromptDTO,
	GetPromptResult,
	PromptResponse,
	DescribePromptResponse,
	DescribePromptDTO,
	DescribeResult,
	ListResult,
	PromptListResponse,
	PromptListDTO,
	PublishPromptDTO,
	PublishPromptResult,
)
from ..utils.protocols import ICache, IApi, ILogger

from ..endpoints.get_prompt import GetPromptEndpoint
from ..endpoints.describe_prompt import DescribePromptEndpoint
from ..endpoints.list_prompts import ListPromptsEndpoint
from ..endpoints.publish_prompt import PublishPromptEndpoint
from ..objects.trace import Trace
from ..objects.generation import Generation
from ..objects.prompt import Prompt
from ..utils.flusher import Flusher
from datetime import datetime

class PromptSDK:
    """
    SDK for interacting with Basalt prompts.
    """
    def __init__(
            self,
            api: IApi,
            cache: ICache,
            fallback_cache: ICache,
            logger: ILogger
        ):
        self._api = api
        self._cache = cache
        self._fallback_cache = fallback_cache

        # Cache responses for 5 minutes
        self._cache_duration = 5 * 60
        self._logger = logger

    async def get(
        self,
        slug: str,
        version: Optional[str] = None,
        tag: Optional[str] = None,
        variables: Optional[Dict[str, str]] = None,
        cache_enabled: bool = True
    ) -> GetPromptResult:
        """
        Retrieve a prompt by slug, optionally specifying version and tag.

        Args:
            slug (str): The slug identifier for the prompt.
            version (Optional[str]): The version of the prompt.
            tag (Optional[str]): The tag associated with the prompt.
            variables (dict): A dictionary of variables to replace in the prompt text.
            cache_enabled (bool): Enable or disable cache for this request.

        Returns:
            Tuple[Optional[Exception], Optional[PromptResponse], Optional[Generation]]:
            A tuple containing an optional exception, an optional PromptResponse, and an optional Generation object.
        """

        dto = GetPromptDTO(
            slug=slug,
            version=version,
            tag=tag
        )

        cached = self._cache.get(dto) if cache_enabled else None

        if cached:
            prompt_response = cast(PromptResponse, cached)
            prompt = self._create_prompt_instance(prompt_response, variables)
            generation = self._prepare_monitoring(prompt)

            return None, prompt, generation

        err, result = await self._api.invoke(GetPromptEndpoint, dto)

        if err is None:
            prompt = self._create_prompt_instance(result.prompt, variables)

            self._cache.put(dto, result.prompt, self._cache_duration)
            self._fallback_cache.put(dto, result.prompt, self._cache_duration)

            generation = self._prepare_monitoring(prompt)

            return err, prompt, generation

        fallback = self._fallback_cache.get(dto) if cache_enabled else None

        if fallback:
            prompt_response = cast(PromptResponse, fallback)
            prompt = self._create_prompt_instance(prompt_response, variables)
            generation = self._prepare_monitoring(prompt)

            return None, prompt, generation

        return err, None, None

    def get_sync(
        self,
        slug: str,
        version: Optional[str] = None,
        tag: Optional[str] = None,
        variables: Optional[Dict[str, str]] = None,
        cache_enabled: bool = True
    ) -> GetPromptResult:
        """
        Retrieve a prompt by slug, optionally specifying version and tag.

        Args:
            slug (str): The slug identifier for the prompt.
            version (Optional[str]): The version of the prompt.
            tag (Optional[str]): The tag associated with the prompt.
            variables (dict): A dictionary of variables to replace in the prompt text.
            cache_enabled (bool): Enable or disable cache for this request.

        Returns:
            Tuple[Optional[Exception], Optional[PromptResponse], Optional[Generation]]:
            A tuple containing an optional exception, an optional PromptResponse, and an optional Generation object.
        """
        dto = GetPromptDTO(
            slug=slug,
            version=version,
            tag=tag
        )

        cached = self._cache.get(dto) if cache_enabled else None

        if cached:
            prompt_response = cast(PromptResponse, cached)
            prompt = self._create_prompt_instance(prompt_response, variables)
            generation = self._prepare_monitoring(prompt)

            return None, prompt, generation

        err, result = self._api.invoke_sync(GetPromptEndpoint, dto)

        if err is None:
            prompt = self._create_prompt_instance(result.prompt, variables)

            self._cache.put(dto, result.prompt, self._cache_duration)
            self._fallback_cache.put(dto, result.prompt, self._cache_duration)

            generation = self._prepare_monitoring(prompt)

            return err, prompt, generation

        fallback = self._fallback_cache.get(dto) if cache_enabled else None

        if fallback:
            prompt_response = cast(PromptResponse, fallback)
            prompt = self._create_prompt_instance(prompt_response, variables)
            generation = self._prepare_monitoring(prompt)

            return None, prompt, generation

        return err, None, None

    def _prepare_monitoring(self,prompt: IPrompt,
    ) -> Generation:
        """
        Prepare monitoring by creating a trace and generation object.

        Args:
            prompt (PromptResponse): The prompt response.
            prompt (Prompt): The prompt

        Returns:
            Generation: The generation object.
        """
        # Create a flusher
        flusher = Flusher(self._api, self._logger)

        # Create a trace
        trace = Trace(prompt.slug, TraceParams(
            input=prompt.text,
            start_time=datetime.now()
        ), flusher, self._logger)

        # Create a generation
        generation = Generation({
            "name": prompt.slug,
            "trace": trace,
            "prompt": {
                "slug": prompt.slug,
                "version": prompt.version,
                "tag": prompt.tag
            },
            "input": prompt.text,
            "variables": prompt.variables,
            "options": {
                "type": "single"
            }
        })

        return generation

    async def describe(
        self,
        slug: str,
        version: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> DescribeResult:
        """
        Get details about a prompt by slug, optionally specifying version and tag.

        Args:
            slug (str): The slug identifier for the prompt.
            version (Optional[str]): The version of the prompt.
            tag (Optional[str]): The tag associated with the prompt.

        Returns:
            Tuple[Optional[Exception], Optional[DescribePromptResponse]]: A tuple containing an optional exception and an optional DescribePromptResponse.
        """
        dto = DescribePromptDTO(
            slug=slug,
            version=version,
            tag=tag
        )

        err, result = await self._api.invoke(DescribePromptEndpoint, dto)

        if err is None:
            prompt = result.prompt

            return None, DescribePromptResponse(
                slug=prompt.slug,
                status=prompt.status,
                name=prompt.name,
                description=prompt.description,
                available_versions=prompt.available_versions,
                available_tags=prompt.available_tags,
                variables=prompt.variables
            )

        return err, None

    def describe_sync(
        self,
        slug: str,
        version: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> DescribeResult:
        """
        Synchronously get details about a prompt by slug, optionally specifying version and tag.

        Args:
            slug (str): The slug identifier for the prompt.
            version (Optional[str]): The version of the prompt.
            tag (Optional[str]): The tag associated with the prompt.

        Returns:
            Tuple[Optional[Exception], Optional[DescribePromptResponse]]: A tuple containing an optional exception and an optional DescribePromptResponse.
        """
        dto = DescribePromptDTO(
            slug=slug,
            version=version,
            tag=tag
        )

        err, result = self._api.invoke_sync(DescribePromptEndpoint, dto)

        if err is None:
            prompt = result.prompt

            return None, DescribePromptResponse(
                slug=prompt.slug,
                status=prompt.status,
                name=prompt.name,
                description=prompt.description,
                available_versions=prompt.available_versions,
                available_tags=prompt.available_tags,
                variables=prompt.variables
            )

        return err, None

    async def list(self, feature_slug: Optional[str] = None) -> ListResult:
        """
        List prompts, optionally filtering by feature_slug.

        Args:
            feature_slug (Optional[str]): Optional feature slug to filter prompts by.

        Returns:
            Tuple[Optional[Exception], Optional[List[PromptListResponse]]]: A tuple containing an optional exception and an optional list of PromptListResponse objects.
        """
        dto = PromptListDTO(featureSlug=feature_slug)

        err, result = await self._api.invoke(ListPromptsEndpoint, dto)

        if err is not None:
            return err, None

        return None, [PromptListResponse(
            slug=prompt.slug,
            status=prompt.status,
            name=prompt.name,
            description=prompt.description,
            available_versions=prompt.available_versions,
            available_tags=prompt.available_tags
        ) for prompt in result.prompts]

    def list_sync(self, feature_slug: Optional[str] = None) -> ListResult:
        """
        Synchronously list prompts, optionally filtering by feature_slug.

        Args:
            feature_slug (Optional[str]): Optional feature slug to filter prompts by.

        Returns:
            Tuple[Optional[Exception], Optional[List[PromptListResponse]]]: A tuple containing an optional exception and an optional list of PromptListResponse objects.
        """
        dto = PromptListDTO(featureSlug=feature_slug)

        err, result = self._api.invoke_sync(ListPromptsEndpoint, dto)

        if err is not None:
            return err, None

        return None, [PromptListResponse(
            slug=prompt.slug,
            status=prompt.status,
            name=prompt.name,
            description=prompt.description,
            available_versions=prompt.available_versions,
            available_tags=prompt.available_tags
        ) for prompt in result.prompts]

    async def publish(
        self,
        slug: str,
        new_tag: str,
        version: Optional[str] = None,
        tag: Optional[str] = None
    ) -> PublishPromptResult:
        """
        Publish a prompt by assigning a tag to a specific version.

        Args:
            slug (str): The slug identifier for the prompt.
            new_tag (str): The new tag to assign to the prompt version.
            version (Optional[str]): The version number to publish.
            tag (Optional[str]): The existing tag to publish.

        Returns:
            Tuple[Optional[Exception], Optional[DeploymentTagResponse]]:
            A tuple containing an optional exception and an optional DeploymentTagResponse.
        """
        if not version and not tag:
            return ValueError("Either version or tag must be provided"), None

        dto = PublishPromptDTO(
            slug=slug,
            new_tag=new_tag,
            version=version,
            tag=tag
        )

        err, result = await self._api.invoke(PublishPromptEndpoint, dto)

        if err is not None:
            return err, None

        return None, result.deploymentTag

    def publish_sync(
        self,
        slug: str,
        new_tag: str,
        version: Optional[str] = None,
        tag: Optional[str] = None
    ) -> PublishPromptResult:
        """
        Synchronously publish a prompt by assigning a tag to a specific version.

        Args:
            slug (str): The slug identifier for the prompt.
            new_tag (str): The new tag to assign to the prompt version.
            version (Optional[str]): The version number to publish.
            tag (Optional[str]): The existing tag to publish.

        Returns:
            Tuple[Optional[Exception], Optional[DeploymentTagResponse]]:
            A tuple containing an optional exception and an optional DeploymentTagResponse.
        """
        if not version and not tag:
            return ValueError("Either version or tag must be provided"), None

        dto = PublishPromptDTO(
            slug=slug,
            new_tag=new_tag,
            version=version,
            tag=tag
        )

        err, result = self._api.invoke_sync(PublishPromptEndpoint, dto)

        if err is not None:
            return err, None

        return None, result.deploymentTag

    @staticmethod
    def _create_prompt_instance(
        prompt_response: PromptResponse,
        variables: Optional[dict] = None
    ) -> Prompt:
        return Prompt(PromptParams(
            slug=prompt_response.slug,
            text=prompt_response.text,
            tag=prompt_response.tag,
            model=prompt_response.model,
            version=prompt_response.version,
            system_text=prompt_response.systemText,
            variables=variables
        ))
