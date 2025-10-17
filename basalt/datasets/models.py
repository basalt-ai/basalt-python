"""
Data models for the Datasets API.

This module contains all data models and data transfer objects used
by the DatasetsClient.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetRowValue:
    """A value in a dataset row."""
    label: str
    value: str


@dataclass
class DatasetRow:
    """
    A row in a dataset.

    Attributes:
        values: Dictionary mapping column names to values.
        name: Optional name for the row.
        ideal_output: Optional ideal output for evaluation.
        metadata: Optional metadata dictionary.
    """
    values: dict[str, str]
    name: str | None = None
    ideal_output: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetRow:
        """Create a DatasetRow from API response dictionary."""
        return cls(
            values=data["values"],
            name=data.get("name"),
            ideal_output=data.get("idealOutput"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Dataset:
    """
    A dataset in the Basalt system.

    Attributes:
        slug: The unique identifier for the dataset.
        name: The human-readable name of the dataset.
        columns: List of column names in the dataset.
        rows: List of rows in the dataset.
    """
    slug: str
    name: str
    columns: list[str]
    rows: list[DatasetRow] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Dataset:
        """Create a Dataset from API response dictionary."""
        rows = []
        if "rows" in data:
            rows = [DatasetRow.from_dict(row) for row in data["rows"]]

        return cls(
            slug=data["slug"],
            name=data["name"],
            columns=data["columns"],
            rows=rows,
        )
