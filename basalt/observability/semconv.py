"""
OpenTelemetry Semantic Conventions for Basalt.

This module defines semantic conventions (attribute names, types, and descriptions)
used for instrumenting Basalt operations with OpenTelemetry.

The conventions are divided into:
1. Standard OpenTelemetry GenAI attributes (gen_ai.*)
2. Standard OpenTelemetry HTTP attributes (http.*)
3. Basalt-specific domain attributes (basalt.*)

References:
- OpenTelemetry GenAI Semantic Conventions:
  https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md
- OpenTelemetry HTTP Semantic Conventions:
  https://github.com/open-telemetry/semantic-conventions/blob/main/docs/http/http-spans.md
"""

from __future__ import annotations

from typing import Final

# ============================================================================
# OpenTelemetry GenAI Semantic Conventions
# ============================================================================
# These attributes follow the official OpenTelemetry semantic conventions
# for Generative AI operations.
# See: https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/


class GenAI:
    """
    OpenTelemetry Generative AI semantic conventions.

    These attributes are used to instrument LLM and GenAI operations
    following the official OpenTelemetry specification.
    """

    # Required attributes
    OPERATION_NAME: Final[str] = "gen_ai.operation.name"
    """
    The name of the operation being performed.
    Type: string
    Required: Yes
    Examples: "chat", "text_completion", "generate_content"
    """

    PROVIDER_NAME: Final[str] = "gen_ai.provider.name"
    """
    The Generative AI provider as identified by the client or server instrumentation.
    Type: string
    Required: Yes
    Examples: "openai", "anthropic", "gcp.vertex_ai", "azure.ai.openai"
    """

    # Request attributes
    REQUEST_MODEL: Final[str] = "gen_ai.request.model"
    """
    The name of the GenAI model a request is being made to.
    Type: string
    Required: Conditionally (if available)
    Examples: "gpt-4", "claude-3-opus-20240229", "gemini-pro"
    """

    REQUEST_TEMPERATURE: Final[str] = "gen_ai.request.temperature"
    """
    The temperature setting for the GenAI request.
    Type: double
    Required: Recommended
    Examples: 0.7, 1.0
    """

    REQUEST_TOP_P: Final[str] = "gen_ai.request.top_p"
    """
    The top_p sampling setting for the GenAI request.
    Type: double
    Required: Recommended
    Examples: 0.9, 1.0
    """

    REQUEST_TOP_K: Final[str] = "gen_ai.request.top_k"
    """
    The top_k sampling setting for the GenAI request.
    Type: double
    Required: Recommended
    Examples: 40.0, 50.0
    """

    REQUEST_MAX_TOKENS: Final[str] = "gen_ai.request.max_tokens"
    """
    The maximum number of tokens the model generates for a request.
    Type: int
    Required: Recommended
    Examples: 100, 2048
    """

    REQUEST_FREQUENCY_PENALTY: Final[str] = "gen_ai.request.frequency_penalty"
    """
    The frequency penalty setting for the GenAI request.
    Type: double
    Required: Recommended
    Examples: 0.1, 0.5
    """

    REQUEST_PRESENCE_PENALTY: Final[str] = "gen_ai.request.presence_penalty"
    """
    The presence penalty setting for the GenAI request.
    Type: double
    Required: Recommended
    Examples: 0.1, 0.5
    """

    REQUEST_STOP_SEQUENCES: Final[str] = "gen_ai.request.stop_sequences"
    """
    List of sequences that the model will use to stop generating further tokens.
    Type: string[]
    Required: Recommended
    Examples: ["forest", "lived"], ["\n"]
    """

    REQUEST_SEED: Final[str] = "gen_ai.request.seed"
    """
    Requests with same seed value more likely to return same result.
    Type: int
    Required: Conditionally (if applicable and if the request includes a seed)
    Examples: 42, 12345
    """

    # Response attributes
    RESPONSE_ID: Final[str] = "gen_ai.response.id"
    """
    The unique identifier for the completion.
    Type: string
    Required: Recommended
    Examples: "chatcmpl-123", "msg_01XFgH7EHk3jn8a"
    """

    RESPONSE_MODEL: Final[str] = "gen_ai.response.model"
    """
    The name of the model that generated the response.
    Type: string
    Required: Recommended
    Examples: "gpt-4-0613", "claude-3-opus-20240229"
    """

    RESPONSE_FINISH_REASONS: Final[str] = "gen_ai.response.finish_reasons"
    """
    Array of reasons the model stopped generating tokens, corresponding to each generation received.
    Type: string[]
    Required: Recommended
    Examples: ["stop"], ["stop", "length"]
    """

    # Usage/Token attributes
    USAGE_INPUT_TOKENS: Final[str] = "gen_ai.usage.input_tokens"
    """
    The number of tokens used in the GenAI input (prompt).
    Type: int
    Required: Recommended
    Examples: 100, 520
    """

    USAGE_OUTPUT_TOKENS: Final[str] = "gen_ai.usage.output_tokens"
    """
    The number of tokens used in the GenAI response (completion).
    Type: int
    Required: Recommended
    Examples: 42, 180
    """

    # Input/Output messages (opt-in, in development)
    INPUT_MESSAGES: Final[str] = "gen_ai.input.messages"
    """
    The chat history provided to the model as an input.
    Type: JSON array of message objects
    Required: Opt-in (in development)
    Note: Replaces deprecated gen_ai.prompt attribute.
    """

    OUTPUT_MESSAGES: Final[str] = "gen_ai.output.messages"
    """
    Messages returned by the model where each message represents a specific model response.
    Type: JSON array of message objects
    Required: Opt-in (in development)
    Note: Replaces deprecated gen_ai.completion attribute.
    """

    # Tool execution attributes
    TOOL_NAME: Final[str] = "gen_ai.tool.name"
    """
    The name of the tool being called.
    Type: string
    Required: Required for tool execution spans
    Examples: "get_weather", "calculate"
    """

    TOOL_KIND: Final[str] = "gen_ai.tool.kind"
    """
    The kind of tool.
    Type: string
    Required: Recommended
    Examples: "function", "retrieval"
    """

    TOOL_CALL_ID: Final[str] = "gen_ai.tool.call.id"
    """
    Unique identifier for the tool call.
    Type: string
    Required: Recommended
    Examples: "call_VSPygqKTWdrhaFErNvMV18Yl"
    """

    TOOL_CALL_ARGUMENTS: Final[str] = "gen_ai.tool.call.arguments"
    """
    The arguments passed to the tool call.
    Type: JSON object
    Required: Recommended
    """

    TOOL_CALL_RESULT: Final[str] = "gen_ai.tool.call.result"
    """
    The result returned from the tool call.
    Type: JSON object
    Required: Recommended
    """


# ============================================================================
# Standard OpenTelemetry HTTP Semantic Conventions
# ============================================================================


class HTTP:
    """
    Standard OpenTelemetry HTTP semantic conventions.

    These follow the official OpenTelemetry HTTP specification.
    See: https://github.com/open-telemetry/semantic-conventions/blob/main/docs/http/
    """

    METHOD: Final[str] = "http.method"
    """
    HTTP request method.
    Type: string
    Examples: "GET", "POST", "PUT"
    """

    URL: Final[str] = "http.url"
    """
    Full HTTP request URL.
    Type: string
    Examples: "https://api.example.com/v1/resource"
    """

    STATUS_CODE: Final[str] = "http.status_code"
    """
    HTTP response status code.
    Type: int
    Examples: 200, 404, 500
    """

    RESPONSE_TIME_MS: Final[str] = "http.response_time_ms"
    """
    HTTP response time in milliseconds.
    Type: float
    Examples: 125.5, 1200.0
    """


# ============================================================================
# Basalt Domain-Specific Semantic Conventions
# ============================================================================
# These are custom attributes specific to Basalt's domain model and features.
# All Basalt attributes use the "basalt." prefix for clarity.


class BasaltSpan:
    """Basalt span metadata attributes."""

    KIND: Final[str] = "basalt.span.kind"
    """
    The kind of Basalt span.
    Type: string
    Examples: "generation", "retrieval", "tool", "event"
    """

    INPUT: Final[str] = "basalt.span.input"
    """
    Canonical input payload associated with the span.
    Type: string (JSON-serialized) or scalar
    Examples: "{\"prompt\": \"hello\"}", "foo=bar"
    """

    OUTPUT: Final[str] = "basalt.span.output"
    """
    Canonical output payload associated with the span.
    Type: string (JSON-serialized) or scalar
    Examples: "{\"answer\": \"42\"}", "done"
    """

    VARIABLES: Final[str] = "basalt.span.variables"
    """
    Optional variables/context dictionary applied to the span.
    Type: string (JSON-serialized)
    Examples: "{\"user_id\": \"123\"}"
    """

    EVALUATORS: Final[str] = "basalt.span.evaluators"
    """
    Names of evaluators attached to the span.
    Type: array of strings
    Examples: ["answer-correctness", "safety"]
    """

    EVALUATION_CONFIG: Final[str] = "basalt.span.evaluation.config"
    """
    Optional, span-scoped configuration applied to evaluators as a whole.
    Type: JSON object (string-serialized) or key/value attributes under this prefix
    Examples: '{"sample_rate": 0.25, "mode": "async"}'
    """

    # Optional prefix for span-scoped evaluator metadata (not per evaluator)
    # Example usage: set attributes like "basalt.span.evaluator.sample_rate" = 0.5
    EVALUATOR_PREFIX: Final[str] = "basalt.span.evaluator"
    """
    Prefix for evaluator-related, span-scoped attributes.
    Type: various (string, number, boolean)
    Examples: "basalt.span.evaluator.sample_rate" = 0.5
    """

    FEATURE_SLUG: Final[str] = "basalt.span.feature_slug"

    METADATA: Final[str] = "basalt.metadata"
    """
    Metadata dictionary for the span.
    Type: string (JSON-serialized)
    """

    IN_TRACE: Final[str] = "basalt.in_trace"
    """
    Indicates whether this span is part of a Basalt observability trace.
    Type: boolean
    Value: true for all spans created by Basalt (start_observe, observe) or
           auto-instrumented spans within a Basalt trace.
    """


class BasaltAPI:
    """Basalt API client operation attributes."""

    CLIENT: Final[str] = "basalt.api.client"
    """
    Logical Basalt API client name.
    Type: string
    Examples: "prompts", "datasets", "experiments"
    """

    OPERATION: Final[str] = "basalt.api.operation"
    """
    Operation name within the Basalt API client.
    Type: string
    Examples: "get", "list", "create", "update"
    """

    INTERNAL: Final[str] = "basalt.internal.api"
    """
    Marks internal Basalt SDK API calls.
    Type: boolean
    Value: True for SDK->API requests (prompts, datasets, etc.)
    """


class BasaltCache:
    """Basalt caching attributes."""

    HIT: Final[str] = "basalt.cache.hit"
    """
    Whether the request was served from cache.
    Type: boolean
    Examples: True, False
    """


class BasaltRequest:
    """Basalt request operation attributes."""

    DURATION_MS: Final[str] = "basalt.request.duration_ms"
    """
    Request duration in milliseconds.
    Type: float
    Examples: 125.5, 1200.0
    """

    SUCCESS: Final[str] = "basalt.request.success"
    """
    Whether the request completed successfully.
    Type: boolean
    Examples: True, False
    """


class BasaltRetrieval:
    """
    Basalt retrieval and vector search operation attributes.

    These are custom to Basalt and cover vector database operations,
    semantic search, and retrieval-augmented generation (RAG).
    """

    QUERY: Final[str] = "basalt.retrieval.query"
    """
    The query text used for retrieval/search.
    Type: string
    Examples: "What is the capital of France?"
    """

    RESULTS_COUNT: Final[str] = "basalt.retrieval.results.count"
    """
    Number of results returned from retrieval operation.
    Type: int
    Examples: 5, 10
    """

    TOP_K: Final[str] = "basalt.retrieval.top_k"
    """
    Top-K parameter for retrieval (number of results requested).
    Type: int
    Examples: 5, 10
    """


class BasaltFunction:
    """Basalt compute/function execution attributes."""

    NAME: Final[str] = "basalt.function.name"
    """
    Logical function or computation name.
    Type: string
    Examples: "generate_embeddings", "score_prompt"
    """

    STAGE: Final[str] = "basalt.function.stage"
    """
    Execution stage or phase within the function.
    Type: string
    Examples: "preprocess", "postprocess"
    """

    METRIC_PREFIX: Final[str] = "basalt.function.metric"
    """
    Prefix for custom function metrics stored as attributes.
    Metrics recorded as ``f"{METRIC_PREFIX}.{key}"``.
    Type: number/string
    Examples: "basalt.function.metric.latency_ms"
    """


class BasaltTool:
    """Basalt tool invocation attributes."""

    NAME: Final[str] = "basalt.tool.name"
    """
    Name of the tool being invoked.
    Type: string
    Examples: "calculator", "web_search"
    """

    INPUT: Final[str] = "basalt.tool.input"
    """
    Input payload sent to the tool.
    Type: string (JSON serialized)
    """

    OUTPUT: Final[str] = "basalt.tool.output"
    """
    Output payload returned from the tool.
    Type: string (JSON serialized)
    """


class BasaltEvent:
    """Basalt custom application event attributes."""

    TYPE: Final[str] = "basalt.event.type"
    """
    Type of custom event.
    Type: string
    Examples: "user_feedback", "validation_error"
    """

    PAYLOAD: Final[str] = "basalt.event.payload"
    """
    Event payload data.
    Type: string (JSON serialized)
    """


class BasaltUser:
    """Basalt user identity attributes."""

    ID: Final[str] = "basalt.user.id"
    """
    Unique user identifier.
    Type: string
    Examples: "user_123", "usr_abc"
    """

    NAME: Final[str] = "basalt.user.name"
    """
    User display name.
    Type: string
    Examples: "John Doe"
    """


class BasaltOrganization:
    """Basalt organization identity attributes."""

    ID: Final[str] = "basalt.organization.id"
    """
    Unique organization identifier.
    Type: string
    Examples: "org_123"
    """

    NAME: Final[str] = "basalt.organization.name"
    """
    Organization display name.
    Type: string
    Examples: "Acme Corp"
    """


class BasaltExperiment:
    """Basalt experiment/feature flag attributes."""

    ID: Final[str] = "basalt.experiment.id"
    """
    Unique experiment identifier.
    Type: string
    Examples: "exp_123"
    """

    NAME: Final[str] = "basalt.experiment.name"
    """
    Experiment display name.
    Type: string
    Examples: "New Prompt Template Test"
    """

    FEATURE_SLUG: Final[str] = "basalt.experiment.feature_slug"
    """
    Feature slug for the experiment.
    Type: string
    Examples: "prompt-v2", "new-model-test"
    """


class BasaltSDK:
    """Basalt SDK metadata attributes."""

    TYPE: Final[str] = "basalt.sdk.type"
    """
    SDK type identifier.
    Type: string
    Examples: "python", "nodejs"
    """

    VERSION: Final[str] = "basalt.sdk.version"
    """
    SDK version.
    Type: string
    Examples: "0.1.0", "1.2.3"
    """


# Metadata prefix for arbitrary key-value pairs
BASALT_META_PREFIX: Final[str] = "basalt.meta."
"""
Prefix for arbitrary metadata attributes.
Usage: basalt.meta.<key>
Type: any
Examples: basalt.meta.session_id, basalt.meta.request_id
"""


# ============================================================================
# Standard OpenTelemetry Resource Attributes
# ============================================================================


class Deployment:
    """Standard OpenTelemetry deployment environment attributes."""

    ENVIRONMENT: Final[str] = "deployment.environment"
    """
    Deployment environment name.
    Type: string
    Examples: "production", "staging", "development"
    """


# ============================================================================
# Deprecated Attributes (for reference only - DO NOT USE)
# ============================================================================


class _DeprecatedGenAI:
    """
    Deprecated GenAI attributes that have been replaced.

    These are kept here for reference only to help with migration.
    DO NOT USE THESE IN NEW CODE.
    """

    # DEPRECATED: Use gen_ai.input.messages or Event API instead
    PROMPT_DEPRECATED: Final[str] = "gen_ai.prompt"
    """
    DEPRECATED: Use gen_ai.input.messages or Event API instead.
    Removed, no replacement at this time.
    """

    # DEPRECATED: Use gen_ai.output.messages or Event API instead
    COMPLETION_DEPRECATED: Final[str] = "gen_ai.completion"
    """
    DEPRECATED: Use gen_ai.output.messages or Event API instead.
    Removed, no replacement at this time.
    """

    # DEPRECATED: Use gen_ai.provider.name instead
    SYSTEM_DEPRECATED: Final[str] = "gen_ai.system"
    """
    DEPRECATED: Use gen_ai.provider.name instead.
    """

    # DEPRECATED: Use gen_ai.usage.input_tokens instead
    USAGE_PROMPT_TOKENS_DEPRECATED: Final[str] = "gen_ai.usage.prompt_tokens"
    """
    DEPRECATED: Use gen_ai.usage.input_tokens instead.
    """

    # DEPRECATED: Use gen_ai.usage.output_tokens instead
    USAGE_COMPLETION_TOKENS_DEPRECATED: Final[str] = "gen_ai.usage.completion_tokens"
    """
    DEPRECATED: Use gen_ai.usage.output_tokens instead.
    """
