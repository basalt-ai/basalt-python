from typing import Optional, Tuple, Dict

from .dtos import GetPromptDTO, PromptResponse
from .protocols import ICache, IApi, ILogger

from .endpoints.get_prompt import GetPromptEndpoint
from .utils import replace_variables

GetResult = Tuple[Optional[Exception], Optional[PromptResponse]]

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
    ) -> GetResult:
        """
        Retrieve a prompt by slug, optionally specifying version and tag.
        
        Args:
            slug (str): The slug identifier for the prompt.
            version (Optional[str]): The version of the prompt.
            tag (Optional[str]): The tag associated with the prompt.
            variables (dict): A dictionnary of variables to replace in the prompt text.
            cache_enabled (bool): Enable or disable cache for this request.
        
        Returns:
            GetResult: A tuple containing an optional exception and an optional PromptResponse.
        """
        dto = GetPromptDTO(
            slug=slug,
            version=version,
            tag=tag
        )
        
        cached = self._cache.get(dto) if cache_enabled else None

        if cached:
            return self._replace_vars(cached, variables)

        err, result = self._api.invoke(GetPromptEndpoint, dto)

        if err is None:
            self._cache.put(dto, result.prompt, duration=self._cache_duration)
            self._fallback_cache.put(dto, result.prompt)

            return self._replace_vars(result.prompt, variables)

        fallback = self._fallback_cache.get(dto) if cache_enabled else None

        if fallback:
            return self._replace_vars(fallback, variables)
        
        return err, None

    def _replace_vars(self, prompt: PromptResponse, variables: Dict[str, str] = {}):
        missing_vars, replaced = replace_variables(prompt.text, variables)

        if missing_vars:
            self._logger.warn(f"""Basalt Warning: Some variables are missing in the prompt text:
    {", ".join(map(str, missing_vars))}""")

        return None, PromptResponse(
            text=replaced,
            model=prompt.model
        )