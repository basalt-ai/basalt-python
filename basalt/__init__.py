"""
Basalt SDK - Python client for the Basalt API.

This module provides the main entry point for the Basalt SDK, including the client
and configuration classes.

Example:
    ```python
    from basalt import Basalt
    from basalt.tracing.provider import BasaltConfig

    # Initialize the client
    config = BasaltConfig(
        service_name="my-app",
        service_version="1.0.0",
        environment="production"
    )
    basalt = Basalt(
        api_key="your-api-key",
        config=config
    )

    # Use the prompts client
    prompt = await basalt.prompts.get("my-prompt")
    ```
"""
from typing import TYPE_CHECKING, Any

from ._version import __version__

# For static analysis / type checkers, expose symbols; at runtime we'll lazily import them.
if TYPE_CHECKING:
    from .client import Basalt  # pragma: no cover
    from .tracing.provider import BasaltConfig  # pragma: no cover
else:
    Basalt: Any = None
    BasaltConfig: Any = None

# Lazily import to avoid importing runtime dependencies (like requests)
# during build-time metadata inspection.
__all__ = ["Basalt", "BasaltConfig", "__version__"]


def __getattr__(name: str):
    if name == "Basalt":
        from .client import Basalt  # imported only when accessed
        return Basalt
    if name == "BasaltConfig":
        from .tracing.provider import BasaltConfig  # imported only when accessed
        return BasaltConfig
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
