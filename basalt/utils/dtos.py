from dataclasses import dataclass
from typing import Any

from ..resources.monitor.experiment_types import Experiment
from ..resources.monitor.generation_types import Generation
from ..resources.prompts.prompt_types import Prompt
from .utils import pick_number, pick_typed


# ------------------------------ Get Prompt ----------------------------- #
@dataclass
class PromptModelParameters:
    temperature: float
    top_k: float
    top_p: float | None

    frequency_penalty: float | None
    presence_penalty: float | None

    max_length: int
    response_format: str
    json_object: dict | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return cls(
            temperature=pick_number(data, "temperature"),
            frequency_penalty=pick_number(data, 'frequencyPenalty') if data.get("frequencyPenalty") else None,
            presence_penalty=pick_number(data, "presencePenalty") if data.get("presencePenalty") else None,
            top_p=pick_number(data, "topP"),
            top_k=pick_number(data, "topK") if data.get("topK") else None,
            max_length=data["maxLength"],
            response_format=pick_typed(data, "responseFormat", str),
            json_object=data.get("jsonObject"),
        )


@dataclass(frozen=True)
class PromptModel:
    provider: str
    model: str
    version: str
    parameters: PromptModelParameters

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return cls(
            provider=pick_typed(data, "provider", str),
            model=pick_typed(data, "model", str),
            version=pick_typed(data, "version", str),
            parameters=PromptModelParameters.from_dict(data.get("parameters")),
        )


@dataclass(frozen=True)
class PromptResponse:
    text: str
    slug: str
    version: str
    tag: str
    model: PromptModel
    systemText: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return cls(
            slug=pick_typed(data, "slug", str),
            tag=pick_typed(data, "tag", str),
            text=pick_typed(data, "text", str),
            model=PromptModel.from_dict(data.get("model")),
            systemText=pick_typed(data, "systemText", str),
            version=pick_typed(data, "version", str),
        )


@dataclass(frozen=True)
class GetPromptDTO:
    slug: str
    tag: str | None = None
    version: str | None = None


GetPromptResult = tuple[Exception | None, Prompt | None, Generation | None]


# ------------------------------ Describe Prompt ----------------------------- #
@dataclass(frozen=True)
class DescribePromptResponse:
    slug: str
    status: str
    name: str
    description: str
    available_versions: list[str]
    available_tags: list[str]
    variables: list[dict[str, str]]

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return cls(
            slug=pick_typed(data, "slug", str) if data.get("slug") else None,
            status=pick_typed(data, "status", str),
            name=pick_typed(data, "name", str),
            description=pick_typed(data, "description", str) if data.get("description") else None,
            available_versions=pick_typed(data, "availableVersions", list),
            available_tags=pick_typed(data, "availableTags", list),
            variables=pick_typed(data, "variables", list),
        )


@dataclass(frozen=True)
class DescribePromptDTO:
    slug: str
    tag: str | None = None
    version: str | None = None


DescribeResult = tuple[Exception | None, DescribePromptResponse | None]


# ------------------------------ List Prompts ----------------------------- #
@dataclass(frozen=True)
class PromptListResponse:
    slug: str
    status: str
    name: str
    description: str
    available_versions: list[str]
    available_tags: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return cls(
            slug=pick_typed(data, "slug", str) if data.get("slug") else None,
            status=pick_typed(data, "status", str),
            name=pick_typed(data, "name", str),
            description=pick_typed(data, "description", str) if data.get("description") else None,
            available_versions=pick_typed(data, "availableVersions", list),
            available_tags=pick_typed(data, "availableTags", list),
        )


@dataclass(frozen=True)
class PromptListDTO:
    featureSlug: str | None = None


ListResult = tuple[Exception | None, list[PromptListResponse] | None]


# ------------------------------ Datasets ----------------------------- #
@dataclass
class DatasetDTO:
    """Dataset data transfer object"""
    slug: str
    name: str
    columns: list[str]
    rows: list['DatasetRowDTO'] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetDTO":
        return cls(
            slug=data["slug"],
            name=data["name"],
            columns=data["columns"],
            rows=[DatasetRowDTO.from_dict(row) for row in data.get("rows", [])]
        )


@dataclass
class DatasetRowDTO:
    """Dataset row data transfer object"""
    values: dict[str, str]
    name: str | None = None
    idealOutput: str | None = None
    metadata: dict[str, Any] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetRowDTO":
        return cls(
            values=data["values"],
            name=data.get("name", None),
            idealOutput=data.get("idealOutput", None),
            metadata=data.get("metadata", {})
        )


@dataclass
class ListDatasetsDTO:
    """DTO for listing datasets"""
    pass


@dataclass
class GetDatasetDTO:
    """DTO for getting a specific dataset"""
    slug: str


@dataclass
class CreateDatasetItemDTO:
    """DTO for creating a dataset item"""
    slug: str
    values: dict[str, str]
    name: str | None = None
    idealOutput: str | None = None
    metadata: dict[str, Any] | None = None


# Result types for dataset operations
ListDatasetsResult = tuple[Exception | None, list[DatasetDTO] | None]
GetDatasetResult = tuple[Exception | None, DatasetDTO | None]
CreateDatasetItemResult = tuple[Exception | None, DatasetRowDTO | None, str | None]

# Result types for monitor operations
CreateExperimentResult = tuple[Exception | None, Experiment | None]
