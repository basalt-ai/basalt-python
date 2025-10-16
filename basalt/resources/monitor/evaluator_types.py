from dataclasses import dataclass
from typing import TypedDict


@dataclass
class Evaluator(TypedDict):
    """
    Represents an evaluator configuration.
    """
    slug: str


@dataclass
class EvaluationConfig(TypedDict, total=False):
    """
    Configuration for the evaluation of the trace and its logs.
    """
    sample_rate: float | None
