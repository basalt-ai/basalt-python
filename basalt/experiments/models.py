"""
Data models for the Experiments API.

This module contains all data models and data transfer objects used
by the ExperimentsClient.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class Experiment:
    """
    An experiment in the Basalt system.

    Immutable and uses slots to reduce per-instance memory overhead.

    Attributes:
        id: The unique identifier for the experiment.
        name: The human-readable name of the experiment.
        feature_slug: The feature slug associated with the experiment.
        created_at: ISO 8601 timestamp of when the experiment was created.
    """
    id: str
    name: str
    feature_slug: str
    created_at: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> Experiment:
        """Create an Experiment from an API response mapping.

        Robust against missing keys or wrong types.

        Args:
            data: The API response data as a mapping.

        Returns:
            An Experiment instance.
        """
        if data is None:
            data = {}

        # Defensive reads with defaults
        id_val = data.get("id") if isinstance(data.get("id"), str) else ""
        name_val = data.get("name") if isinstance(data.get("name"), str) else ""
        feature_slug_val = data.get("featureSlug") if isinstance(data.get("featureSlug"), str) else ""
        created_at_val = data.get("createdAt") if isinstance(data.get("createdAt"), str) else ""

        return cls(
            id=str(id_val),
            name=str(name_val),
            feature_slug=str(feature_slug_val),
            created_at=str(created_at_val),
        )
