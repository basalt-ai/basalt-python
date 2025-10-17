"""
Adapter for PromptsClient to maintain backward compatibility with PromptSDK.

This adapter wraps the new PromptsClient and provides the old tuple-based interface.

.. deprecated:: 0.5.0
    The tuple-based API (error, result) is deprecated. Use the new PromptsClient
    directly which raises exceptions instead of returning error tuples.
    See the migration guide for details.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from ...prompts.models import Prompt as NewPrompt
from ...utils.dtos import (
    DescribeResult,
    GetPromptResult,
    ListResult,
    Prompt,
)
from ...utils.protocols import IApi, ILogger, IPromptSDK

if TYPE_CHECKING:
    from ...prompts.client import PromptsClient


class PromptSDKAdapter(IPromptSDK):
    """
    Adapter that wraps PromptsClient and provides the old PromptSDK interface.

    This maintains backward compatibility by converting exceptions back to tuple returns.
    """

    def __init__(self, client: PromptsClient, api: IApi, logger: ILogger):
        """
        Initialize the adapter.

        Args:
            client: The new PromptsClient instance
            api: API instance (for monitoring/generation support)
            logger: Logger instance

        .. deprecated:: 0.5.0
            This adapter provides backward compatibility for the tuple-based API.
            Use PromptsClient directly for the new exception-based API.
        """
        self._client = client
        self._api = api
        self._logger = logger
        self._warned = False

    async def get(
        self,
        slug: str,
        version: str | None = None,
        tag: str | None = None,
        variables: dict[str, str] | None = None,
        cache_enabled: bool = True
    ) -> GetPromptResult:
        """
        Async get prompt with tuple return for backward compatibility.

        .. deprecated:: 0.5.0
            Tuple-based error handling is deprecated. Use PromptsClient.get() directly
            which raises exceptions instead of returning tuples.

        Returns:
            Tuple[Exception | None, Prompt | None, Generation | None]
        """
        self._emit_deprecation_warning()
        try:
            new_prompt, generation = await self._client.get(
                slug=slug,
                version=version,
                tag=tag,
                variables=variables,
                cache_enabled=cache_enabled
            )
            # Convert new Prompt dataclass back to old Prompt object
            old_prompt = self._convert_prompt(new_prompt)
            return None, old_prompt, generation
        except Exception as e:
            return e, None, None

    def get_sync(
        self,
        slug: str,
        version: str | None = None,
        tag: str | None = None,
        variables: dict[str, str] | None = None,
        cache_enabled: bool = True
    ) -> GetPromptResult:
        """
        Sync get prompt with tuple return for backward compatibility.

        .. deprecated:: 0.5.0
            Tuple-based error handling is deprecated. Use PromptsClient.get_sync() directly
            which raises exceptions instead of returning tuples.

        Returns:
            Tuple[Exception | None, Prompt | None, Generation | None]
        """
        self._emit_deprecation_warning()
        try:
            new_prompt, generation = self._client.get_sync(
                slug=slug,
                version=version,
                tag=tag,
                variables=variables,
                cache_enabled=cache_enabled
            )
            # Convert new Prompt dataclass back to old Prompt object
            old_prompt = self._convert_prompt(new_prompt)
            return None, old_prompt, generation
        except Exception as e:
            return e, None, None

    async def describe(
        self,
        slug: str,
        version: str | None = None,
        tag: str | None = None,
    ) -> DescribeResult:
        """
        Async describe prompt with tuple return.

        Returns:
            Tuple[Exception | None, DescribePromptResponse | None]
        """
        try:
            result = await self._client.describe(slug=slug, version=version, tag=tag)
            return None, result
        except Exception as e:
            return e, None

    def describe_sync(
        self,
        slug: str,
        version: str | None = None,
        tag: str | None = None,
    ) -> DescribeResult:
        """
        Sync describe prompt with tuple return.

        Returns:
            Tuple[Exception | None, DescribePromptResponse | None]
        """
        try:
            result = self._client.describe_sync(slug=slug, version=version, tag=tag)
            return None, result
        except Exception as e:
            return e, None

    async def list(self, feature_slug: str | None = None) -> ListResult:
        """
        Async list prompts with tuple return.

        Returns:
            Tuple[Exception | None, List[PromptListResponse] | None]
        """
        try:
            results = await self._client.list(feature_slug=feature_slug)
            return None, results
        except Exception as e:
            return e, None

    def list_sync(self, feature_slug: str | None = None) -> ListResult:
        """
        Sync list prompts with tuple return.

        Returns:
            Tuple[Exception | None, List[PromptListResponse] | None]
        """
        try:
            results = self._client.list_sync(feature_slug=feature_slug)
            return None, results
        except Exception as e:
            return e, None

    def _emit_deprecation_warning(self):
        """Emit deprecation warning once per adapter instance."""
        if not self._warned:
            warnings.warn(
                "The tuple-based API (error, result) is deprecated and will be removed in v1.0.0. "
                "Use PromptsClient directly for exception-based error handling. "
                "See https://docs.getbasalt.ai/migration-guide for details.",
                DeprecationWarning,
                stacklevel=3
            )
            self._warned = True

    @staticmethod
    def _convert_prompt(new_prompt: NewPrompt) -> Prompt:
        """
        Convert new Prompt dataclass to old Prompt object.

        Args:
            new_prompt: The new Prompt dataclass from PromptsClient

        Returns:
            Old Prompt object compatible with existing code
        """
        from ...objects.prompt import Prompt as PromptObject
        from ...resources.prompts.prompt_types import PromptParams

        # Create old-style Prompt object
        prompt_obj = PromptObject(PromptParams(
            slug=new_prompt.slug,
            text=new_prompt.raw_text,  # Use raw_text to preserve templates
            tag=new_prompt.tag,
            model=new_prompt.model,
            version=new_prompt.version,
            system_text=new_prompt.raw_system_text,
            variables=new_prompt.variables
        ))

        return prompt_obj
