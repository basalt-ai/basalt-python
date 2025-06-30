"""
Dataset types module for Basalt SDK
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class DatasetRowValue:
    """
    A value in a dataset row
    """
    label: str
    value: str


@dataclass
class DatasetRow:
    """
    A row in a dataset
    """
    values: List[DatasetRowValue]
    name: Optional[str] = None
    idealOutput: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetRow":
        """Create a DatasetRow from a dictionary"""
        values_list = []
        if "values" in data:
            if isinstance(data["values"], list):
                values_list = [DatasetRowValue(**val) if isinstance(val, dict) else DatasetRowValue(label=val["label"], value=val["value"]) 
                               for val in data["values"]]
            elif isinstance(data["values"], dict):
                values_list = [DatasetRowValue(label=key, value=val) for key, val in data["values"].items()]
                
        return cls(
            values=values_list,
            name=data.get("name", None),
            idealOutput=data.get("idealOutput", None),
            metadata=data.get("metadata", {})
        )


@dataclass
class Dataset:
    """
    A dataset in the Basalt system
    """
    slug: str
    name: str
    columns: List[str]
    rows: List[DatasetRow] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Dataset":
        """Create a Dataset from a dictionary"""
        rows = []
        if "rows" in data:
            rows = [DatasetRow.from_dict(row) for row in data["rows"]]
            
        return cls(
            slug=data["slug"],
            name=data["name"],
            columns=data["columns"],
            rows=rows
        )
