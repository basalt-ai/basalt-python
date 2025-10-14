from dataclasses import dataclass
from datetime import datetime
from typing import TypedDict


@dataclass
class ExperimentParams(TypedDict):
    """Parameters for creating an experiment."""
    name: str


@dataclass
class Experiment:
    id: str
    name: str
    feature_slug: str
    created_at: datetime
