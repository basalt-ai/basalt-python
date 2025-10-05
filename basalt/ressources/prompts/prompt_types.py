"""
Prompt types module for Basalt SDK
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class PromptModelParameters:
    """Model parameters for a prompt"""
    temperature: float
    max_length: int
    response_format: str
    top_k: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    json_object: Optional[dict] = None


@dataclass
class PromptModel:
    """Model configuration for a prompt"""
    provider: str
    model: str
    version: str
    parameters: PromptModelParameters


@dataclass
class PromptParams:
    """Parameters for creating a new prompt"""
    slug: str
    text: str
    model: PromptModel
    version: str
    system_text: Optional[str] = None
    tag: Optional[str] = None
    variables: Optional[Dict[str, Any]] = None


@dataclass
class Prompt:
    """
    Prompt class representing a prompt template in the Basalt system.

    This class represents a prompt template that can be used for AI model generations.

    Example:
        ```python
        # Get a prompt
        error, prompt = basalt.prompts.get(
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
    system_text: Optional[str] = None
    raw_system_text: Optional[str] = None
    variables: Optional[Dict[str, Any]] = None
    tag: Optional[str] = None

    def compile_variables(self, variables: Dict[str, Any]) -> 'Prompt':
        """
        Compile the prompt variables and render the text and system_text templates.

        Args:
            variables (Dict[str, Any]): A dictionary of variables to render into the prompt templates.

        Returns:
            Prompt: The updated Prompt instance with rendered text and system_text.
        """
        ...
