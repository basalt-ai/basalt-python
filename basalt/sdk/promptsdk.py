"""Deprecated module: PromptSDK has been removed.

Use the feature-based prompt clients and the new tracing/observability APIs instead.

This module intentionally raises an ImportError to prevent usage.
"""

raise ImportError(
    "basalt.sdk.promptsdk is deprecated and removed. Use basalt.prompts.* clients and basalt.observability."
)
