"""
Data models for the Prompts API.

This module contains all data models and data transfer objects used
by the PromptsClient.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class PromptModelParameters:
    """Model parameters for a prompt.

    Immutable and uses slots to reduce per-instance memory overhead.
    """
    temperature: float
    max_length: int
    response_format: str
    top_k: float | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    json_object: dict | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> PromptModelParameters:
        """Create instance from API response mapping.

        Robust against missing keys or wrong types.
        """
        if data is None:
            data = {}

        # Defensive reads with defaults and type checking
        temperature = data.get("temperature")
        temperature = float(temperature) if isinstance(temperature, (int, float)) else 0.0

        max_length = data.get("maxLength")
        max_length = int(max_length) if isinstance(max_length, (int, float)) else 0

        response_format = data.get("responseFormat")
        response_format = str(response_format) if isinstance(response_format, str) else "text"

        top_k = data.get("topK")
        top_k = float(top_k) if isinstance(top_k, (int, float)) else None

        top_p = data.get("topP")
        top_p = float(top_p) if isinstance(top_p, (int, float)) else None

        frequency_penalty = data.get("frequencyPenalty")
        frequency_penalty = float(frequency_penalty) if isinstance(frequency_penalty, (int, float)) else None

        presence_penalty = data.get("presencePenalty")
        presence_penalty = float(presence_penalty) if isinstance(presence_penalty, (int, float)) else None

        json_object = data.get("jsonObject")
        json_object = dict(json_object) if isinstance(json_object, Mapping) else None

        return cls(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_length=max_length,
            response_format=response_format,
            json_object=json_object,
        )


@dataclass(slots=True, frozen=True)
class PromptModel:
    """Model configuration for a prompt.

    Immutable and uses slots to reduce per-instance memory overhead.
    """
    provider: str
    model: str
    version: str
    parameters: PromptModelParameters

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> PromptModel:
        """Create instance from API response mapping.

        Robust against missing keys or wrong types.
        """
        if data is None:
            data = {}

        # Defensive reads with defaults
        provider = data.get("provider") if isinstance(data.get("provider"), str) else ""
        model = data.get("model") if isinstance(data.get("model"), str) else ""
        version = data.get("version") if isinstance(data.get("version"), str) else ""

        parameters_data = data.get("parameters")
        parameters = PromptModelParameters.from_dict(parameters_data if isinstance(parameters_data, Mapping) else None)

        return cls(
            provider=str(provider),
            model=str(model),
            version=str(version),
            parameters=parameters,
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


class PromptContextManager:
    """
    Context manager wrapper for Prompt that enables observability.

    This class wraps a Prompt dataclass and:
    1. Always creates a span for prompt fetches (even in imperative usage)
    2. Optionally extends span lifetime when used as context manager to scope GenAI calls
    3. Forwards all attribute access to the wrapped Prompt transparently

    Usage:
        # Imperative (span created and immediately ended):
        prompt = prompts.get_sync("slug")

        # Context manager (span stays open to scope GenAI calls):
        with prompts.get_sync("slug") as prompt:
            response = llm.generate(prompt.text)
    """

    def __init__(
        self,
        prompt: Prompt,
        slug: str,
        version: str | None,
        tag: str | None,
        variables: dict[str, Any] | None,
        from_cache: bool = False,
    ):
        """
        Initialize the wrapper and create an immediate span for tracking.

        Args:
            prompt: The Prompt dataclass to wrap
            slug: Prompt slug for span naming and attributes
            version: Prompt version for span attributes
            tag: Prompt tag for span attributes
            variables: Variables used in prompt compilation
            from_cache: Whether the prompt was retrieved from cache
        """
        self._prompt = prompt
        self._slug = slug
        self._version = version
        self._tag = tag
        self._variables = variables
        self._from_cache = from_cache

        # Context manager state
        self._span = None
        self._span_context = None
        self._tracer = None

        # Create and immediately end a span for imperative usage tracking
        self._create_immediate_span()

    def _create_immediate_span(self):
        """Create a span that immediately ends to track prompt fetches."""
        try:
            from ..observability.context_managers import get_tracer

            tracer = get_tracer("basalt.prompts")

            # Create span context
            with tracer.start_as_current_span(f"prompt.{self._slug}") as span:
                # Set span attributes
                self._set_span_attributes(span)
        except Exception:
            # Silently fail if observability is not available
            # Prompt functionality should not break due to observability issues
            pass

    def _set_span_attributes(self, span):
        """Set all prompt-related attributes on a span."""
        import json

        span.set_attribute("basalt.span.kind", "function")
        span.set_attribute("basalt.prompt.slug", self._slug)

        if self._version:
            span.set_attribute("basalt.prompt.version", self._version)

        if self._tag:
            span.set_attribute("basalt.prompt.tag", self._tag)

        span.set_attribute("basalt.prompt.model.provider", self._prompt.model.provider)
        span.set_attribute("basalt.prompt.model.model", self._prompt.model.model)

        if self._variables:
            span.set_attribute("basalt.prompt.variables", json.dumps(self._variables))

        span.set_attribute("basalt.prompt.from_cache", self._from_cache)

    def __getattr__(self, name):
        """Forward all attribute access to the wrapped Prompt."""
        return getattr(self._prompt, name)

    def __enter__(self):
        """
        Enter context manager mode - create a new span that stays open.

        This allows auto-instrumented GenAI calls to nest under the prompt span.
        """
        try:
            from opentelemetry import trace

            from ..observability.context_managers import get_tracer

            self._tracer = get_tracer("basalt.prompts")

            # Start a new span that will stay open
            self._span = self._tracer.start_span(f"prompt.{self._slug}")
            self._set_span_attributes(self._span)

            # Use span context that will auto-end on exit
            self._span_context = trace.use_span(self._span, end_on_exit=True)
            self._span_context.__enter__()
        except Exception:
            # Silently fail - observability should not break functionality
            pass

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager mode - end the span."""
        if self._span_context:
            try:
                self._span_context.__exit__(exc_type, exc_val, exc_tb)
            except Exception:
                pass

        # Don't suppress exceptions
        return False

    def __repr__(self):
        """Return representation forwarded from wrapped Prompt."""
        return repr(self._prompt)

    def __str__(self):
        """Return string representation forwarded from wrapped Prompt."""
        return str(self._prompt)


class AsyncPromptContextManager:
    """
    Async context manager wrapper for Prompt that enables observability.

    This is the async version of PromptContextManager for use with
    `async with prompts.get(...) as prompt:` syntax.
    """

    def __init__(
        self,
        prompt: Prompt,
        slug: str,
        version: str | None,
        tag: str | None,
        variables: dict[str, Any] | None,
        from_cache: bool = False,
    ):
        """
        Initialize the wrapper and create an immediate span for tracking.

        Args:
            prompt: The Prompt dataclass to wrap
            slug: Prompt slug for span naming and attributes
            version: Prompt version for span attributes
            tag: Prompt tag for span attributes
            variables: Variables used in prompt compilation
            from_cache: Whether the prompt was retrieved from cache
        """
        self._prompt = prompt
        self._slug = slug
        self._version = version
        self._tag = tag
        self._variables = variables
        self._from_cache = from_cache

        # Context manager state
        self._span = None
        self._span_context = None
        self._tracer = None

        # Create and immediately end a span for imperative usage tracking
        self._create_immediate_span()

    def _create_immediate_span(self):
        """Create a span that immediately ends to track prompt fetches."""
        try:
            from ..observability.context_managers import get_tracer

            tracer = get_tracer("basalt.prompts")

            # Create span context
            with tracer.start_as_current_span(f"prompt.{self._slug}") as span:
                # Set span attributes
                self._set_span_attributes(span)
        except Exception:
            # Silently fail if observability is not available
            pass

    def _set_span_attributes(self, span):
        """Set all prompt-related attributes on a span."""
        import json

        span.set_attribute("basalt.span.kind", "function")
        span.set_attribute("basalt.prompt.slug", self._slug)

        if self._version:
            span.set_attribute("basalt.prompt.version", self._version)

        if self._tag:
            span.set_attribute("basalt.prompt.tag", self._tag)

        span.set_attribute("basalt.prompt.model.provider", self._prompt.model.provider)
        span.set_attribute("basalt.prompt.model.model", self._prompt.model.model)

        if self._variables:
            span.set_attribute("basalt.prompt.variables", json.dumps(self._variables))

        span.set_attribute("basalt.prompt.from_cache", self._from_cache)

    def __getattr__(self, name):
        """Forward all attribute access to the wrapped Prompt."""
        return getattr(self._prompt, name)

    async def __aenter__(self):
        """
        Enter async context manager mode - create a new span that stays open.

        This allows auto-instrumented GenAI calls to nest under the prompt span.
        """
        try:
            from opentelemetry import trace

            from ..observability.context_managers import get_tracer

            self._tracer = get_tracer("basalt.prompts")

            # Start a new span that will stay open
            self._span = self._tracer.start_span(f"prompt.{self._slug}")
            self._set_span_attributes(self._span)

            # Use span context that will auto-end on exit
            self._span_context = trace.use_span(self._span, end_on_exit=True)
            self._span_context.__enter__()
        except Exception:
            # Silently fail - observability should not break functionality
            pass

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager mode - end the span."""
        if self._span_context:
            try:
                self._span_context.__exit__(exc_type, exc_val, exc_tb)
            except Exception:
                pass

        # Don't suppress exceptions
        return False

    def __repr__(self):
        """Return representation forwarded from wrapped Prompt."""
        return repr(self._prompt)

    def __str__(self):
        """Return string representation forwarded from wrapped Prompt."""
        return str(self._prompt)


@dataclass(slots=True, frozen=True)
class PromptResponse:
    """Response from the Get Prompt API endpoint.

    Immutable and uses slots to reduce per-instance memory overhead.
    """
    text: str
    slug: str
    version: str
    model: PromptModel
    system_text: str
    tag: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> PromptResponse:
        """Create instance from API response mapping.

        Robust against missing keys or wrong types.
        """
        if data is None:
            data = {}

        slug = data.get("slug") if isinstance(data.get("slug"), str) else ""
        tag = data.get("tag") if isinstance(data.get("tag"), str) else None
        text = data.get("text") if isinstance(data.get("text"), str) else ""
        system_text = data.get("systemText") if isinstance(data.get("systemText"), str) else ""
        version = data.get("version") if isinstance(data.get("version"), str) else ""

        model_data = data.get("model")
        model = PromptModel.from_dict(model_data if isinstance(model_data, Mapping) else None)

        return cls(
            slug=str(slug),
            tag=tag,
            text=str(text),
            model=model,
            system_text=str(system_text),
            version=str(version),
        )


@dataclass(slots=True, frozen=True)
class DescribePromptResponse:
    """Response from the Describe Prompt API endpoint.

    Immutable and uses slots to reduce per-instance memory overhead.
    """
    slug: str
    status: str
    name: str
    description: str
    available_versions: list[str]
    available_tags: list[str]
    variables: list[dict[str, str]]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> DescribePromptResponse:
        """Create instance from API response mapping.

        Robust against missing keys or wrong types. Copies mutable inputs.
        """
        if data is None:
            data = {}

        slug = data.get("slug") if isinstance(data.get("slug"), str) else ""
        status = data.get("status") if isinstance(data.get("status"), str) else ""
        name = data.get("name") if isinstance(data.get("name"), str) else ""
        description = data.get("description") if isinstance(data.get("description"), str) else ""

        available_versions_raw = data.get("availableVersions")
        available_versions = list(available_versions_raw) if isinstance(available_versions_raw, list) else []

        available_tags_raw = data.get("availableTags")
        available_tags = list(available_tags_raw) if isinstance(available_tags_raw, list) else []

        variables_raw = data.get("variables")
        variables = list(variables_raw) if isinstance(variables_raw, list) else []

        return cls(
            slug=str(slug),
            status=str(status),
            name=str(name),
            description=str(description),
            available_versions=available_versions,
            available_tags=available_tags,
            variables=variables,
        )


@dataclass(slots=True, frozen=True)
class PromptListResponse:
    """Response item from the List Prompts API endpoint.

    Immutable and uses slots to reduce per-instance memory overhead.
    """
    slug: str
    status: str
    name: str
    description: str
    available_versions: list[str]
    available_tags: list[str]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> PromptListResponse:
        """Create instance from API response mapping.

        Robust against missing keys or wrong types. Copies mutable inputs.
        """
        if data is None:
            data = {}

        slug = data.get("slug") if isinstance(data.get("slug"), str) else ""
        status = data.get("status") if isinstance(data.get("status"), str) else ""
        name = data.get("name") if isinstance(data.get("name"), str) else ""
        description = data.get("description") if isinstance(data.get("description"), str) else ""

        available_versions_raw = data.get("availableVersions")
        available_versions = list(available_versions_raw) if isinstance(available_versions_raw, list) else []

        available_tags_raw = data.get("availableTags")
        available_tags = list(available_tags_raw) if isinstance(available_tags_raw, list) else []

        return cls(
            slug=str(slug),
            status=str(status),
            name=str(name),
            description=str(description),
            available_versions=available_versions,
            available_tags=available_tags,
        )


@dataclass(slots=True, frozen=True)
class PublishPromptResponse:
    """Response from the Publish Prompt API endpoint.

    Immutable and uses slots to reduce per-instance memory overhead.
    """
    id: str
    label: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> PublishPromptResponse:
        """Create instance from API response mapping.

        Robust against missing keys or wrong types.
        """
        if data is None:
            data = {}

        # Handle nested deploymentTag structure
        deployment_tag = data.get("deploymentTag")
        if isinstance(deployment_tag, Mapping):
            id_val = deployment_tag.get("id") if isinstance(deployment_tag.get("id"), str) else ""
            label_val = deployment_tag.get("label") if isinstance(deployment_tag.get("label"), str) else ""
        else:
            id_val = data.get("id") if isinstance(data.get("id"), str) else ""
            label_val = data.get("label") if isinstance(data.get("label"), str) else ""

        return cls(
            id=str(id_val),
            label=str(label_val),
        )
