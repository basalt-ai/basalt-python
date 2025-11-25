"""
Data models for the Datasets API.

This module contains all data models and data transfer objects used
by the DatasetsClient.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class DatasetColumn:
    """Definition of a dataset column.

    Immutable and uses slots to reduce per-instance memory overhead for
    large datasets.
    """

    name: str
    type: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | str) -> DatasetColumn:
        """Create a DatasetColumn from API response dictionary or a simple
        column-name string.
        """
        if isinstance(data, str):
            return cls(name=data, type=None)

        # defensive reads with defaults
        name = data.get("name") if isinstance(data.get("name"), str) else ""
        col_type = data.get("type") if isinstance(data.get("type"), str) else None
        return cls(name=str(name), type=col_type)

@dataclass(slots=True)
class DatasetRow:
    """
    A row in a dataset.

    Attributes:
        values: Dictionary mapping column names to values.
        name: Optional name for the row.
        ideal_output: Optional ideal output for evaluation.
        metadata: Optional metadata dictionary.
    """
    # store as a plain dict internally for fast access; accept Mapping inputs
    values: dict[str, str]
    name: str | None = None
    ideal_output: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # defensive copies to avoid aliasing mutable inputs
        # ensure values and metadata are real dicts
        # Always create defensive copies to avoid aliasing mutable inputs.
        # This ensures that even when callers pass a dict, we don't hold a
        # reference to the same object.
        try:
            self.values = dict(self.values)
        except (TypeError, ValueError):
            # Not convertible to dict (e.g., None), fall back to empty dict
            self.values = {}

        try:
            self.metadata = dict(self.metadata)
        except (TypeError, ValueError):
            self.metadata = {}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DatasetRow:
        """Create a DatasetRow from API response mapping.

        Robust against missing keys or wrong types. Copies mutable inputs.
        """
        if data is None:
            data = {}

        # tolerate non-mapping 'values' by attempting to convert
        raw_values = data.get("values") if isinstance(data, Mapping) else None
        if not isinstance(raw_values, Mapping):
            # fallback: empty dict
            values = {}
        else:
            values = dict(raw_values)

        name = data.get("name") if isinstance(data.get("name"), str) else None
        ideal = data.get("idealOutput") if isinstance(data.get("idealOutput"), str) else None
        metadata_raw = data.get("metadata")
        metadata = dict(metadata_raw) if isinstance(metadata_raw, Mapping) else {}

        return cls(values=values, name=name, ideal_output=ideal, metadata=metadata)


@dataclass(slots=True)
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
    # public attributes; rows kept mutable, columns are immutable objects
    columns: list[DatasetColumn] = field(default_factory=list)
    rows: list[DatasetRow] = field(default_factory=list)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | None,
    ) -> Dataset:
        """Create a Dataset from an API response mapping.

        Accepts columns as either a list of strings or list of mappings. Returns
        immutable `DatasetColumn` objects for columns and defensive copies of
        rows and their contents.
        """
        if data is None:
            data = {}

        # columns can be a list of mappings or strings
        columns_raw = data.get("columns") if isinstance(data, Mapping) else []
        column_definitions: list[DatasetColumn] = []
        if isinstance(columns_raw, Iterable):
            for col in columns_raw:
                try:
                    column_definitions.append(DatasetColumn.from_dict(col))
                except (TypeError, AttributeError, ValueError):
                    continue

        # rows: parse and create defensive copies
        rows_raw = data.get("rows") if isinstance(data, Mapping) else None
        rows_list: list[DatasetRow] = []

        if isinstance(rows_raw, Iterable):
            for r in rows_raw:
                try:
                    rows_list.append(DatasetRow.from_dict(r))
                except (TypeError, AttributeError, ValueError):
                    # skip malformed rows
                    continue

        slug_val = data.get("slug") if isinstance(data.get("slug"), str) else ""
        name_val = data.get("name") if isinstance(data.get("name"), str) else ""

        return cls(slug=str(slug_val), name=str(name_val), columns=column_definitions, rows=rows_list)
