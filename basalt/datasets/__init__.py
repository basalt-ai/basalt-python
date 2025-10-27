"""
Datasets package for the Basalt SDK.

This package exposes the DatasetsClient and related models for interacting
with the Basalt Datasets API. The client is lazily imported to avoid
module-level circular dependencies during initialization.
"""
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from .client import DatasetsClient

from .models import Dataset, DatasetRow

__all__ = ["DatasetsClient", "Dataset", "DatasetRow"]


def __getattr__(name: str) -> Any:
    if name == "DatasetsClient":
        from .client import DatasetsClient

        return DatasetsClient
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
