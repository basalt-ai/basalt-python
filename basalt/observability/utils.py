"""Utility functions for Basalt observability."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from opentelemetry.trace import Span

if TYPE_CHECKING:
    from .context_managers import LLMSpanHandle

from . import semconv
from .context_managers import trace_content_enabled
from .trace_context import TraceIdentity


def apply_span_metadata(span: Span, metadata: Mapping[str, Any] | None) -> None:
    """Apply metadata to a span as an aggregated JSON object at basalt.metadata.

    Behavior:
    - Shallow merge with any existing metadata from prior calls.
    - Metadata is stored as a JSON-serialized dictionary at ``semconv.BasaltSpan.METADATA``.
    - Values that cannot be JSON-serialized will fallback to ``str(value)``.
    """
    if not metadata:
        return

    # Collect existing metadata from prior JSON blob
    existing: dict[str, Any] = {}
    attributes = getattr(span, "attributes", None)
    if isinstance(attributes, Mapping):
        prior_blob = attributes.get(semconv.BasaltSpan.METADATA)
        if isinstance(prior_blob, str):
            try:
                parsed = json.loads(prior_blob)
                if isinstance(parsed, dict):
                    existing = parsed
            except Exception:
                pass

    # Perform shallow merge (incoming overrides existing)
    merged = {**existing, **metadata}

    # Update aggregated JSON attribute
    try:
        span.set_attribute(semconv.BasaltSpan.METADATA, json.dumps(merged))
    except Exception:
        # Fallback: try to serialize with str() for non-serializable values
        try:
            safe_merged = {
                k: v if isinstance(v, (str, bool, int, float, type(None))) else str(v)
                for k, v in merged.items()
            }
            span.set_attribute(semconv.BasaltSpan.METADATA, json.dumps(safe_merged))
        except Exception:
            pass


def apply_prompt_context_attributes(span: Span, prompt_ctx: Mapping[str, Any]) -> None:
    span.set_attribute("basalt.prompt.slug", prompt_ctx["slug"])
    if prompt_ctx.get("version"):
        span.set_attribute("basalt.prompt.version", prompt_ctx["version"])
    if prompt_ctx.get("tag"):
        span.set_attribute("basalt.prompt.tag", prompt_ctx["tag"])
    span.set_attribute("basalt.prompt.model.provider", prompt_ctx["provider"])
    span.set_attribute("basalt.prompt.model.model", prompt_ctx["model"])
    if prompt_ctx.get("variables"):
        span.set_attribute("basalt.prompt.variables", json.dumps(prompt_ctx["variables"]))
    span.set_attribute("basalt.prompt.from_cache", prompt_ctx["from_cache"])


def resolve_attributes(
    attributes: object,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> object:
    """Resolve attributes into a dictionary."""
    if attributes is None:
        return None
    if callable(attributes):
        try:
            return attributes(*args, **kwargs)
        except Exception:
            return None
    return attributes


def resolve_bound_arguments(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> inspect.BoundArguments | None:
    """Bind arguments to function signature."""
    try:
        signature = inspect.signature(func)
        return signature.bind_partial(*args, **kwargs)
    except Exception:
        return None


def resolve_payload_from_bound(
    resolver: object,
    bound: inspect.BoundArguments | None,
) -> object:
    """Resolve input payload from bound arguments."""
    if resolver is None:
        if not bound:
            return None
        return dict(bound.arguments)
    if callable(resolver):
        return resolver(bound)
    if isinstance(resolver, str):
        if not bound:
            return None
        return bound.arguments.get(resolver)
    if isinstance(resolver, Sequence) and not isinstance(resolver, (str, bytes)):
        if not bound:
            return None
        return {
            name: bound.arguments.get(name)
            for name in resolver
            if bound.arguments.get(name) is not None
        }
    return resolver


def resolve_variables_payload(
    resolver: dict[str, Any]
    | Callable[[inspect.BoundArguments | None], Mapping[str, Any] | None]
    | Sequence[str]
    | Mapping[str, Any]
    | None,
    bound: inspect.BoundArguments | None,
) -> Mapping[str, Any] | None:
    """Resolve variables payload."""
    if resolver is None:
        return None
    if callable(resolver):
        payload = resolver(bound)
        if payload is None:
            return None
        if not isinstance(payload, Mapping):
            return None
        return payload
    if isinstance(resolver, Mapping):
        return {str(key): value for key, value in resolver.items()}
    if isinstance(resolver, Sequence) and not isinstance(resolver, (str, bytes)):
        if not bound:
            return None
        return {
            name: bound.arguments.get(name)
            for name in resolver
            if bound.arguments.get(name) is not None
        }
    return None


def resolve_evaluators_payload(
    resolver: object,
    bound: inspect.BoundArguments | None,
    result: object | None = None,
) -> list[Any] | None:
    """Resolve evaluator specifications."""
    if resolver is None:
        return None
    if callable(resolver):
        try:
            value = resolver(bound, result)
        except TypeError:
            value = resolver(bound)
    else:
        value = resolver
    if value is None:
        return None
    if isinstance(value, (str, Mapping)):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def _normalize_identity_value(
    value: object,
) -> TraceIdentity | dict[str, Any] | None:
    """Normalize a user/org identity specification."""
    if value is None:
        return None
    if isinstance(value, TraceIdentity):
        return value
    if isinstance(value, Mapping):
        normalized = dict(value)
        identifier = normalized.get("id")
        if isinstance(identifier, (int, float)):
            normalized["id"] = str(identifier)
        return normalized
    if isinstance(value, (str, int, float)):
        identifier = str(value).strip()
        if not identifier:
            return None
        return {"id": identifier}
    raise TypeError("Identity payload must be a mapping, TraceIdentity, or primitive identifier.")


def resolve_identity_payload(
    resolver: object,
    bound: inspect.BoundArguments | None,
) -> tuple[TraceIdentity | dict[str, Any] | None, TraceIdentity | dict[str, Any] | None]:
    """
    Resolve identity information into user and organization payloads.

    The resolver can be:
        - None
        - A mapping with ``user``/``organization`` keys
        - A sequence of [user, organization]
        - A callable returning one of the above
        - A primitive id or TraceIdentity interpreted as the user identity
    """

    if resolver is None:
        return None, None

    payload = resolver(bound) if callable(resolver) else resolver
    if payload is None:
        return None, None

    def _from_mapping(
        mapping: Mapping[str, Any],
    ) -> tuple[TraceIdentity | dict[str, Any] | None, TraceIdentity | dict[str, Any] | None]:
        lowered = {
            str(key).lower(): value for key, value in mapping.items() if isinstance(key, str)
        }

        user_spec: object | None = None
        org_spec: object | None = None

        if "user" in lowered:
            user_spec = lowered["user"]
        elif "user_id" in lowered:
            user_spec = {"id": lowered["user_id"], "name": lowered.get("user_name")}
        elif "userid" in lowered:
            user_spec = {"id": lowered["userid"]}

        if "organization" in lowered:
            org_spec = lowered["organization"]
        elif "org" in lowered:
            org_spec = lowered["org"]
        else:
            org_identifier = lowered.get("organization_id") or lowered.get("org_id")
            if org_identifier is not None:
                org_spec = {
                    "id": org_identifier,
                    "name": lowered.get("organization_name") or lowered.get("org_name"),
                }

        if user_spec is None and org_spec is None:
            user_spec = mapping

        return _normalize_identity_value(user_spec), _normalize_identity_value(org_spec)

    if isinstance(payload, Mapping):
        return _from_mapping(payload)

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        items = list(payload)
        user_spec = items[0] if items else None
        org_spec = items[1] if len(items) > 1 else None
        return _normalize_identity_value(user_spec), _normalize_identity_value(org_spec)

    return _normalize_identity_value(payload), None


def _extract_first(bound, keys: tuple[str, ...]) -> object:
    if not bound:
        return None
    for key in keys:
        if key in bound.arguments:
            return bound.arguments[key]
    return None


def default_generation_input(bound: inspect.BoundArguments | None) -> object:
    value = _extract_first(bound, ("prompt", "input", "inputs", "messages", "question"))
    if value is not None:
        return value
    return dict(bound.arguments) if bound else None


def default_generation_variables(bound: inspect.BoundArguments | None) -> Mapping[str, Any] | None:
    value = _extract_first(bound, ("variables", "params", "context"))
    return value if isinstance(value, Mapping) else None


def default_retrieval_input(bound: inspect.BoundArguments | None) -> object:
    value = _extract_first(bound, ("query", "question", "text", "search"))
    if value is not None:
        return value
    return dict(bound.arguments) if bound else None


def default_retrieval_variables(bound: inspect.BoundArguments | None) -> Mapping[str, Any] | None:
    value = _extract_first(bound, ("filters", "metadata", "options"))
    return value if isinstance(value, Mapping) else None


def serialize_prompt(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except Exception:
        return str(value)


def extract_completion(result: object) -> str | None:
    if result is None:
        return None
    if isinstance(result, str):
        return result

    data: dict[str, Any] | None = None
    if isinstance(result, dict):
        data = result
    elif hasattr(result, "model_dump"):
        try:
            # Using getattr for type-safe dynamic attribute access
            model_dump = result.model_dump # type: ignore[attr-defined]
            data = model_dump()
        except Exception:
            data = None
    elif hasattr(result, "dict"):
        try:
            # Using getattr for type-safe dynamic attribute access
            dict_method = result.dict # type: ignore[attr-defined]
            data = dict_method()
        except Exception:
            data = None
    elif hasattr(result, "__dict__"):
        data = vars(result)

    if not data:
        return None

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
            text = choice.get("text")
            if isinstance(text, str):
                return text

    completion = data.get("completion") or data.get("output")
    if isinstance(completion, str):
        return completion
    return None


def extract_usage(result: object) -> tuple[int | None, int | None]:
    usage_section: object | None = None
    if isinstance(result, dict):
        usage_section = result.get("usage")
    elif hasattr(result, "usage"):
        # Using getattr for type-safe dynamic attribute access
        usage_section = getattr(result, "usage", None)
    else:
        model_dump = getattr(result, "model_dump", None)
        if callable(model_dump):
            try:
                dumped = model_dump()
                usage_section = dumped.get("usage") if isinstance(dumped, dict) else None
            except Exception:
                usage_section = None
    if not isinstance(usage_section, dict):
        return None, None
    input_tokens = usage_section.get("prompt_tokens") or usage_section.get("input_tokens")
    output_tokens = usage_section.get("completion_tokens") or usage_section.get("output_tokens")
    input_tokens = int(input_tokens) if isinstance(input_tokens, (int, float)) else None
    output_tokens = int(output_tokens) if isinstance(output_tokens, (int, float)) else None
    return input_tokens, output_tokens


def apply_llm_request_metadata(span: LLMSpanHandle, bound: inspect.BoundArguments | None) -> None:
    if not bound:
        return
    model = _extract_first(bound, ("model", "model_name"))
    if isinstance(model, str):
        span.set_model(model)
    prompt = _extract_first(bound, ("prompt", "input", "inputs", "messages", "question"))
    serialized = serialize_prompt(prompt)
    if serialized and trace_content_enabled():
        span.set_prompt(serialized)


def apply_llm_response_metadata(span: LLMSpanHandle, result: object) -> None:
    completion = extract_completion(result)
    if completion and trace_content_enabled():
        span.set_completion(completion)
    input_tokens, output_tokens = extract_usage(result)
    if input_tokens is not None or output_tokens is not None:
        span.set_tokens(input=input_tokens, output=output_tokens)
