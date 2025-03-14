from typing import Optional, Dict, Tuple, Any

from ..utils.dtos import GetPromptDTO, PromptResponse, DescribePromptResponse, DescribePromptDTO, GetResult, DescribeResult, ListResult, PromptListResponse
from ..utils.protocols import ICache, IApi, ILogger

from ..endpoints.get_prompt import GetPromptEndpoint
from ..endpoints.describe_prompt import DescribePromptEndpoint
from ..endpoints.list_prompts import ListPromptsEndpoint
from ..utils.utils import replace_variables
from ..objects.trace import Trace
from ..objects.generation import Generation
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

    def get(
        self,
        slug: str,
        version: Optional[str] = None,
        tag: Optional[str] = None,
        variables: Dict[str, str] = {},
        cache_enabled: bool = True
    ) -> Tuple[Optional[Exception], Optional[PromptResponse], Optional[Generation]]:
        """
        Retrieve a prompt by slug, optionally specifying version and tag.

        Args:
            slug (str): The slug identifier for the prompt.
            version (Optional[str]): The version of the prompt.
            tag (Optional[str]): The tag associated with the prompt.
            variables (dict): A dictionnary of variables to replace in the prompt text.
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
            original_prompt_text = cached.text
            err, prompt_response = self._replace_vars(cached, variables)
            generation = self._prepare_monitoring(prompt_response, slug, version, tag, variables, original_prompt_text)
            return err, prompt_response, generation

        err, result = self._api.invoke(GetPromptEndpoint, dto)

        if err is None:
            original_prompt_text = result.prompt.text
            self._cache.put(dto, result.prompt, ttl=self._cache_duration)
            self._fallback_cache.put(dto, result.prompt)

            err, prompt_response = self._replace_vars(result.prompt, variables)
            generation = self._prepare_monitoring(prompt_response, slug, version, tag, variables, original_prompt_text)
            return err, prompt_response, generation

        fallback = self._fallback_cache.get(dto) if cache_enabled else None

        if fallback:
            original_prompt_text = fallback.text
            err, prompt_response = self._replace_vars(fallback, variables)
            generation = self._prepare_monitoring(prompt_response, slug, version, tag, variables, original_prompt_text)
            return err, prompt_response, generation

        return err, None, None

    def _prepare_monitoring(
        self, 
        prompt: PromptResponse, 
        slug: str, 
        version: Optional[str] = None, 
        tag: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
        original_prompt_text: Optional[str] = None
    ) -> Generation:
        """
        Prepare monitoring by creating a trace and generation object.
        
        Args:
            prompt (PromptResponse): The prompt response.
            slug (str): The slug identifier for the prompt.
            version (Optional[str]): The version of the prompt.
            tag (Optional[str]): The tag associated with the prompt.
            variables (Optional[Dict[str, Any]]): Variables used in the prompt.
            original_prompt_text (Optional[str]): The original prompt text.
            
        Returns:
            Generation: The generation object.
        """
        # Create a flusher
        flusher = Flusher(self._api, self._logger)
        
        # Create a trace
        trace = Trace(slug, {
            "input": original_prompt_text or prompt.text,
            "start_time": datetime.now()
        }, flusher)
        
        # Create a generation
        generation = Generation({
            "name": slug,
            "trace": trace,
            "prompt": {
                "slug": slug,
                "version": version,
                "tag": tag
            },
            "input": original_prompt_text or prompt.text,
            "variables": variables,
            "options": {"type": "single"}
        })
        
        return generation

    def describe(
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
            cache_enabled (bool): Enable or disable cache for this request.

        Returns:
            Tuple[Optional[Exception], Optional[DescribePromptResponse]]: A tuple containing an optional exception and an optional DescribePromptResponse.
        """
        dto = DescribePromptDTO(
            slug=slug,
            version=version,
            tag=tag
        )

        err, result = self._api.invoke(DescribePromptEndpoint, dto)

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

    def list(self) -> ListResult:
        err, result = self._api.invoke(ListPromptsEndpoint)

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

    def _replace_vars(self, prompt: PromptResponse, variables: Dict[str, str] = {}):
        missing_vars, replaced = replace_variables(prompt.text, variables)

        if missing_vars:
            self._logger.warn(f"""Basalt Warning: Some variables are missing in the prompt text:
    {", ".join(map(str, missing_vars))}""")

        return None, PromptResponse(
            text=replaced,
            model=prompt.model
        )