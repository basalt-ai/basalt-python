from dataclasses import dataclass
from datetime import datetime
from typing import Any


# Minimal Experiment model (expand as needed)
@dataclass
class Experiment:
    feature_slug: str
    name: str
    id: str
    created_at: datetime

    # Add more fields as needed

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return cls(
            feature_slug=data.get("featureSlug") or data.get("feature_slug"),
            name=data.get("name"),
            id=data.get("id"),
            created_at=data.get("createdAt"),
        )


@dataclass
class CreateExperimentDTO:
    feature_slug: str
    name: str


@dataclass
class Output:
    experiment: Experiment


class CreateExperimentEndpoint:
    """
    Endpoint for creating an experiment
    """

    @staticmethod
    def prepare_request(dto: CreateExperimentDTO) -> dict[str, Any]:
        body = {
            "featureSlug": dto.feature_slug,
            "name": dto.name,
        }

        return {
            "method": "post",
            "path": "/monitor/experiments",
            "body": body,
        }

    @staticmethod
    def decode_response(body: Any) -> tuple[Exception | None, Output | None]:
        if not isinstance(body, dict):
            return Exception("Failed to decode response (invalid body format)"), None

        experiment = Experiment.from_dict(body)
        return None, Output(experiment=experiment)
