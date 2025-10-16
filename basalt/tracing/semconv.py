"""Semantic conventions for Basalt SDK tracing.

This module defines semantic conventions (standard naming for attributes, span names, etc.)
specific to the Basalt SDK. These conventions help ensure consistency across traces
and make it easier to query and analyze telemetry data.

For general OpenTelemetry semantic conventions, see:
https://opentelemetry.io/docs/specs/semconv/
"""

# Span Attributes
# ---------------
# These constants define attribute keys for span attributes

# Basalt-specific span kind (e.g., "prompt_fetch", "dataset_operation", "trace_send")
BASALT_SPAN_KIND = "basalt.span.kind"

# Operation type (e.g., "fetch", "create", "update", "delete")
BASALT_OPERATION_TYPE = "basalt.operation.type"

# Resource identifiers
BASALT_PROMPT_ID = "basalt.prompt.id"
BASALT_PROMPT_NAME = "basalt.prompt.name"
BASALT_PROMPT_VERSION = "basalt.prompt.version"
BASALT_DATASET_ID = "basalt.dataset.id"
BASALT_DATASET_NAME = "basalt.dataset.name"
BASALT_TRACE_ID = "basalt.trace.id"
BASALT_EXPERIMENT_ID = "basalt.experiment.id"

# API-related attributes
BASALT_API_ENDPOINT = "basalt.api.endpoint"
BASALT_API_METHOD = "basalt.api.method"
BASALT_API_STATUS_CODE = "basalt.api.status_code"

# Cache-related attributes
BASALT_CACHE_HIT = "basalt.cache.hit"
BASALT_CACHE_KEY = "basalt.cache.key"

# SDK metadata
BASALT_SDK_VERSION = "basalt.sdk.version"
BASALT_SDK_TYPE = "basalt.sdk.type"

# Error attributes
BASALT_ERROR_TYPE = "basalt.error.type"
BASALT_ERROR_MESSAGE = "basalt.error.message"

# Span Kinds
# ----------
# Standard values for the BASALT_SPAN_KIND attribute

class SpanKind:
    """Standard span kind values for Basalt operations."""

    # Prompt operations
    PROMPT_FETCH = "prompt.fetch"
    PROMPT_LIST = "prompt.list"
    PROMPT_DESCRIBE = "prompt.describe"
    PROMPT_RENDER = "prompt.render"

    # Dataset operations
    DATASET_FETCH = "dataset.fetch"
    DATASET_LIST = "dataset.list"
    DATASET_CREATE_ITEM = "dataset.create_item"

    # Monitoring operations
    TRACE_SEND = "trace.send"
    LOG_SEND = "log.send"
    EXPERIMENT_CREATE = "experiment.create"
    GENERATION_RECORD = "generation.record"

    # HTTP operations
    HTTP_REQUEST = "http.request"

    # Cache operations
    CACHE_GET = "cache.get"
    CACHE_SET = "cache.set"


class OperationType:
    """Standard operation type values."""

    FETCH = "fetch"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    LIST = "list"
    SEND = "send"
    RENDER = "render"


# Span Names
# ----------
# Helper functions to generate consistent span names

def span_name_for_prompt(operation: str, prompt_name: str | None = None) -> str:
    """
    Generate a consistent span name for prompt operations.

    Args:
        operation: The operation type (e.g., "fetch", "list", "describe").
        prompt_name: Optional prompt name to include in the span name.

    Returns:
        A formatted span name.

    Example:
        >>> span_name_for_prompt("fetch", "my-prompt")
        'basalt.prompt.fetch my-prompt'
    """
    if prompt_name:
        return f"basalt.prompt.{operation} {prompt_name}"
    return f"basalt.prompt.{operation}"


def span_name_for_dataset(operation: str, dataset_name: str | None = None) -> str:
    """
    Generate a consistent span name for dataset operations.

    Args:
        operation: The operation type (e.g., "fetch", "list", "create_item").
        dataset_name: Optional dataset name to include in the span name.

    Returns:
        A formatted span name.

    Example:
        >>> span_name_for_dataset("fetch", "my-dataset")
        'basalt.dataset.fetch my-dataset'
    """
    if dataset_name:
        return f"basalt.dataset.{operation} {dataset_name}"
    return f"basalt.dataset.{operation}"


def span_name_for_trace(operation: str) -> str:
    """
    Generate a consistent span name for trace operations.

    Args:
        operation: The operation type (e.g., "send").

    Returns:
        A formatted span name.

    Example:
        >>> span_name_for_trace("send")
        'basalt.trace.send'
    """
    return f"basalt.trace.{operation}"


def span_name_for_http(method: str, endpoint: str) -> str:
    """
    Generate a consistent span name for HTTP requests.

    Args:
        method: The HTTP method (e.g., "GET", "POST").
        endpoint: The API endpoint path.

    Returns:
        A formatted span name.

    Example:
        >>> span_name_for_http("GET", "/api/prompts/my-prompt")
        'HTTP GET /api/prompts/my-prompt'
    """
    return f"HTTP {method} {endpoint}"


# Event Names
# -----------
# Standard event names for span events

class EventName:
    """Standard event names for span events."""

    CACHE_HIT = "cache.hit"
    CACHE_MISS = "cache.miss"
    API_REQUEST_START = "api.request.start"
    API_REQUEST_COMPLETE = "api.request.complete"
    ERROR_OCCURRED = "error.occurred"
    RETRY_ATTEMPT = "retry.attempt"
