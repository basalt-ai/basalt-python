"""
Basalt SDK - Python client for the Basalt API.

This module provides the main entry point for the Basalt SDK, including the client
and configuration classes.

Example:
    ```python
    from basalt import Basalt, TelemetryConfig

    telemetry = TelemetryConfig(service_name="my-app", environment="production")
    basalt = Basalt(api_key="your-api-key", telemetry_config=telemetry)
    prompt = await basalt.prompts.get("my-prompt")
    ```
"""
from typing import TYPE_CHECKING

from ._version import __version__

# For static analysis / type checkers, expose symbols; at runtime we'll lazily import them.
if TYPE_CHECKING:
    from .client import Basalt  # pragma: no cover
    from .observability.config import TelemetryConfig  # pragma: no cover

# Lazily import to avoid importing runtime dependencies (like requests)
# during build-time metadata inspection.
__all__ = ["Basalt", "TelemetryConfig", "__version__"]


def __getattr__(name: str):
    if name == "Basalt":
        from .client import Basalt  # imported only when accessed
        return Basalt
    if name == "TelemetryConfig":
        from .observability.config import TelemetryConfig  # imported only when accessed
        return TelemetryConfig
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
