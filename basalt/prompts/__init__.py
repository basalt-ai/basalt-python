"""
Prompts package for the Basalt SDK.

This package exposes the PromptsClient and related models for interacting
with the Basalt Prompts API. The client is lazily imported to avoid
circular imports during initialization.
"""
from typing import TYPE_CHECKING, Any

from .models import (
    AsyncPromptContextManager,
    DescribePromptResponse,
    Prompt,
    PromptContextManager,
    PromptListResponse,
    PromptModel,
    PromptModelParameters,
    PromptParams,
    PromptResponse,
    PublishPromptResponse,
)

if TYPE_CHECKING:  # pragma: no cover
    from .client import PromptsClient

__all__ = [
    "PromptsClient",
    "Prompt",
    "PromptContextManager",
    "AsyncPromptContextManager",
    "PromptModel",
    "PromptModelParameters",
    "PromptParams",
    "PromptResponse",
    "DescribePromptResponse",
    "PromptListResponse",
    "PublishPromptResponse",
]


def __getattr__(name: str) -> Any:
    if name == "PromptsClient":
        from .client import PromptsClient

        return PromptsClient
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
