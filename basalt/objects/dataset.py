"""
Dataset object for Basalt SDK
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class DatasetRow:
    """
    A row in a dataset with values and metadata
    """
    values: Dict[str, str] 
    name: Optional[str] = None
    ideal_output: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the DatasetRow to a dictionary for API requests"""
        result = {
            "values": self.values,
            "metadata": self.metadata,
			"name": self.name,
			"idealOutput": self.ideal_output
        }
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetRow":
        """
        Create a DatasetRow instance from a dictionary
        
        Args:
            data: Dictionary containing dataset row data
            
        Returns:
            DatasetRow: A new DatasetRow instance
        """
        return cls(
            values=data.get("values", {}),
            name=data.get("name", None),
            ideal_output=data.get("idealOutput", None),
            metadata=data.get("metadata", {})
        )


@dataclass
class Dataset:
    """
    A dataset with rows and metadata
    """
    slug: str
    name: str
    columns: List[str]
    rows: List[DatasetRow] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Dataset to a dictionary for API responses"""
        return {
            "slug": self.slug,
            "name": self.name,
            "columns": self.columns,
            "rows": [row.to_dict() for row in self.rows]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Dataset":
        """
        Create a Dataset instance from a dictionary
        
        Args:
            data: Dictionary containing dataset data
            
        Returns:
            Dataset: A new Dataset instance
        """
        rows = [DatasetRow.from_dict(row) for row in data.get("rows", [])]
        
        return cls(
            slug=data["slug"],
            name=data["name"],
            columns=data["columns"],
            rows=rows
        )
