from dataclasses import dataclass
from typing import Optional

@dataclass
class Evaluator:
    """
    Represents an evaluator configuration.
    """
    slug: str

@dataclass
class EvaluationConfig:
    """
    Configuration for the evaluation of the trace and its logs.
    """
    sample_rate: Optional[float] = None
