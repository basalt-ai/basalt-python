from dataclasses import dataclass
from datetime import datetime

@dataclass
class ExperimentParams:
    """Parameters for creating an experiment."""
    name: str

@dataclass
class Experiment:
    id: str
    name: str
    feature_slug: str
    created_at: datetime
