"""
Semantic conventions for GenAI instrumentation.

This module defines attribute names following OpenTelemetry semantic conventions
for Generative AI operations.

Based on: https://opentelemetry.io/docs/specs/semconv/gen-ai/
"""

# GenAI System Attributes
GEN_AI_SYSTEM = "gen_ai.system"  # The Generative AI system (e.g., "openai", "anthropic")
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"  # The name of the model requested
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"  # The actual model that generated the response

# GenAI Operation Attributes
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"  # The name of the operation (e.g., "chat", "completion")

# GenAI Request Attributes
GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
GEN_AI_REQUEST_TOP_K = "gen_ai.request.top_k"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_REQUEST_FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
GEN_AI_REQUEST_PRESENCE_PENALTY = "gen_ai.request.presence_penalty"
GEN_AI_REQUEST_STOP_SEQUENCES = "gen_ai.request.stop_sequences"

# GenAI Response Attributes
GEN_AI_RESPONSE_ID = "gen_ai.response.id"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"

# GenAI Usage Attributes
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

# GenAI Prompt and Completion (for observability)
GEN_AI_PROMPT = "gen_ai.prompt"  # The user's prompt/input
GEN_AI_COMPLETION = "gen_ai.completion"  # The model's response/completion

# OpenAI-specific attributes
OPENAI_API_VERSION = "openai.api_version"
OPENAI_API_BASE = "openai.api_base"
OPENAI_API_TYPE = "openai.api_type"


class GenAIAttributes:
    """
    Helper class for working with GenAI semantic conventions.

    Example:
        ```python
        attributes = GenAIAttributes.create_request_attributes(
            system="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=1000
        )
        ```
    """

    @staticmethod
    def create_request_attributes(
        system: str,
        model: str,
        operation: str = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop_sequences: list[str] | None = None,
    ) -> dict[str, str | int | float | list]:
        """
        Create a dictionary of GenAI request attributes.

        Args:
            system: The GenAI system name (e.g., "openai")
            model: The model name
            operation: The operation name (default: "chat")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            stop_sequences: Stop sequences

        Returns:
            Dictionary of attributes to attach to a span
        """
        attrs: dict[str, str | int | float | list] = {
            GEN_AI_SYSTEM: system,
            GEN_AI_REQUEST_MODEL: model,
            GEN_AI_OPERATION_NAME: operation,
        }

        if temperature is not None:
            attrs[GEN_AI_REQUEST_TEMPERATURE] = temperature
        if max_tokens is not None:
            attrs[GEN_AI_REQUEST_MAX_TOKENS] = max_tokens
        if top_p is not None:
            attrs[GEN_AI_REQUEST_TOP_P] = top_p
        if top_k is not None:
            attrs[GEN_AI_REQUEST_TOP_K] = top_k
        if frequency_penalty is not None:
            attrs[GEN_AI_REQUEST_FREQUENCY_PENALTY] = frequency_penalty
        if presence_penalty is not None:
            attrs[GEN_AI_REQUEST_PRESENCE_PENALTY] = presence_penalty
        if stop_sequences is not None:
            attrs[GEN_AI_REQUEST_STOP_SEQUENCES] = stop_sequences

        return attrs

    @staticmethod
    def create_response_attributes(
        response_id: str | None = None,
        response_model: str | None = None,
        finish_reasons: list[str] | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> dict[str, str | int | list]:
        """
        Create a dictionary of GenAI response attributes.

        Args:
            response_id: The response ID
            response_model: The actual model used for the response
            finish_reasons: Reasons why generation finished
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens generated

        Returns:
            Dictionary of attributes to attach to a span
        """
        attrs: dict[str, str | int | list] = {}

        if response_id is not None:
            attrs[GEN_AI_RESPONSE_ID] = response_id
        if response_model is not None:
            attrs[GEN_AI_RESPONSE_MODEL] = response_model
        if finish_reasons is not None:
            attrs[GEN_AI_RESPONSE_FINISH_REASONS] = finish_reasons
        if input_tokens is not None:
            attrs[GEN_AI_USAGE_INPUT_TOKENS] = input_tokens
        if output_tokens is not None:
            attrs[GEN_AI_USAGE_OUTPUT_TOKENS] = output_tokens

        return attrs
