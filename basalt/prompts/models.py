"""
Data models for the Prompts API.

This module contains all Pydantic models and data transfer objects used
by the PromptsClient.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PromptModelParameters:
    """Model parameters for a prompt."""
    temperature: float
    max_length: int
    response_format: str
    top_k: float | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    json_object: dict | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptModelParameters:
        """Create instance from API response dictionary."""
        return cls(
            temperature=data.get("temperature", 0.0),
            top_k=data.get("topK"),
            top_p=data.get("topP"),
            frequency_penalty=data.get("frequencyPenalty"),
            presence_penalty=data.get("presencePenalty"),
            max_length=data.get("maxLength", 0),
            response_format=data.get("responseFormat", "text"),
            json_object=data.get("jsonObject"),
        )


@dataclass
class PromptModel:
    """Model configuration for a prompt."""
    provider: str
    model: str
    version: str
    parameters: PromptModelParameters

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptModel:
        """Create instance from API response dictionary."""
        return cls(
            provider=data["provider"],
            model=data["model"],
            version=data["version"],
            parameters=PromptModelParameters.from_dict(data.get("parameters", {})),
        )


@dataclass
class PromptParams:
    """Parameters for creating a new prompt instance."""
    slug: str
    text: str
    model: PromptModel
    version: str
    system_text: str | None = None
    tag: str | None = None
    variables: dict[str, Any] | None = None


@dataclass
class Prompt:
    """
    Prompt class representing a prompt template in the Basalt system.

    This class represents a prompt template that can be used for AI model generations.

    Example:
        ```python
        # Get a prompt
        prompt = basalt.prompts.get(
            slug="qa-prompt",
            version="2.1.0",
            variables={"context": "Paris is the capital of France"}
        )

        # Access prompt properties
        print(prompt.text)
        print(prompt.model.provider)
        ```
    """
    slug: str
    text: str
    raw_text: str
    model: PromptModel
    version: str
    system_text: str | None = None
    raw_system_text: str | None = None
    variables: dict[str, Any] | None = None
    tag: str | None = None

    def compile_variables(self, variables: dict[str, Any]) -> Prompt:
        """
        Compile the prompt variables and render the text and system_text templates.

        Args:
            variables: A dictionary of variables to render into the prompt templates.

        Returns:
            The updated Prompt instance with rendered text and system_text.
        """
        from jinja2 import Template

        self.variables = variables
        self.text = Template(self.raw_text).render(variables)

        if self.raw_system_text:
            self.system_text = Template(self.raw_system_text).render(variables)

        return self


@dataclass
class PromptResponse:
    """Response from the Get Prompt API endpoint."""
    text: str
    slug: str
    version: str
    tag: str
    model: PromptModel
    system_text: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptResponse:
        """Create instance from API response dictionary."""
        return cls(
            slug=data["slug"],
            tag=data["tag"],
            text=data["text"],
            model=PromptModel.from_dict(data["model"]),
            system_text=data.get("systemText", ""),
            version=data["version"],
        )


@dataclass
class DescribePromptResponse:
    """Response from the Describe Prompt API endpoint."""
    slug: str
    status: str
    name: str
    description: str
    available_versions: list[str]
    available_tags: list[str]
    variables: list[dict[str, str]]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DescribePromptResponse:
        """Create instance from API response dictionary."""
        return cls(
            slug=data.get("slug", ""),
            status=data["status"],
            name=data["name"],
            description=data.get("description", ""),
            available_versions=data.get("availableVersions", []),
            available_tags=data.get("availableTags", []),
            variables=data.get("variables", []),
        )


@dataclass
class PromptListResponse:
    """Response item from the List Prompts API endpoint."""
    slug: str
    status: str
    name: str
    description: str
    available_versions: list[str]
    available_tags: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptListResponse:
        """Create instance from API response dictionary."""
        return cls(
            slug=data.get("slug", ""),
            status=data["status"],
            name=data["name"],
            description=data.get("description", ""),
            available_versions=data.get("availableVersions", []),
            available_tags=data.get("availableTags", []),
        )
