"""Decorator utilities for Basalt observability."""

from __future__ import annotations

import functools
import inspect
import json
import random
import warnings
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, TypeAlias, TypeVar

from opentelemetry.trace import StatusCode

from .context_managers import (
    LLMSpanHandle,
    trace_content_enabled,
    with_evaluators,
)
from .context_managers import (
    trace_event as trace_event_cm,
)
from .context_managers import (
    trace_function as trace_function_cm,
)
from .context_managers import (
    trace_generation as trace_generation_cm,
)
from .context_managers import (
    trace_retrieval as trace_retrieval_cm,
)
from .context_managers import (
    trace_span as trace_span_cm,
)
from .context_managers import (
    trace_tool as trace_tool_cm,
)

F = TypeVar("F", bound=Callable[..., Any])
AsyncFunc = TypeVar("AsyncFunc", bound=Callable[..., Awaitable[Any]])

# Type alias for span attributes: static dict, dynamic callable, or None
AttributeSpec: TypeAlias = dict[str, Any] | Callable[..., dict[str, Any]] | None


def _resolve_attributes(
    attributes: AttributeSpec,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Resolve attributes into a dictionary based on their type.

    This function handles different types of attribute specifications:
    - If attributes is None, returns None.
    - If attributes is callable, attempts to call it with the provided args and kwargs,
        returning the result as a dict or None if an exception occurs.
    - Otherwise, assumes attributes is already a dict and returns it directly.

    Parameters
    ----------
    attributes : AttributeSpec
            The attribute specification, which can be None, a callable, or a dict.
    args : tuple[Any, ...]
            Positional arguments to pass to the callable if attributes is callable.
    kwargs : dict[str, Any]
            Keyword arguments to pass to the callable if attributes is callable.

    Returns
    -------
    dict[str, Any] | None
            A dictionary of resolved attributes, or None if resolution fails or is not applicable.
    """
    if attributes is None:
        return None
    if callable(attributes):
        try:
            return attributes(*args, **kwargs)
        except Exception:
            return None
    return attributes


def _bind_args(func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]):
    """
    Bind the provided args and kwargs to the function's signature.
    Returns the BoundArguments object or None if binding fails.
    """
    try:
        signature = inspect.signature(func)
        return signature.bind_partial(*args, **kwargs)
    except Exception:
        return None


def _resolve_bound_arguments(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> inspect.BoundArguments | None:
    """
    Bind the provided args and kwargs to the function's signature.
    Returns the BoundArguments object or None if binding fails.
    """
    try:
        signature = inspect.signature(func)
        return signature.bind_partial(*args, **kwargs)
    except Exception:
        return None


def _resolve_payload_from_bound(
    resolver: Any,
    bound: inspect.BoundArguments | None,
) -> Any:
    """
    Resolve input payload from the bound arguments using the provided resolver.
    """
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


def _resolve_variables_payload(
    resolver: Any,
    bound: inspect.BoundArguments | None,
) -> Mapping[str, Any] | None:
    """
    Resolve variables payload from the bound arguments using the provided resolver.
    """
    if resolver is None:
        return None
    if callable(resolver):
        payload = resolver(bound)
        if payload is None:
            return None
        if not isinstance(payload, Mapping):
            raise TypeError("Variables resolver must return a mapping.")
        return payload
    if isinstance(resolver, Mapping):
        return resolver
    if isinstance(resolver, Sequence) and not isinstance(resolver, (str, bytes)):
        if not bound:
            return None
        return {
            name: bound.arguments.get(name)
            for name in resolver
            if bound.arguments.get(name) is not None
        }
    raise TypeError("Unsupported variables resolver type.")


def _resolve_evaluators_payload(
    resolver: Any,
    bound: inspect.BoundArguments | None,
    result: Any | None = None,
) -> list[Any] | None:
    """
    Resolve evaluator specifications from the bound arguments using the provided resolver.
    """
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


def _wrap_with_span(
    span_factory: Callable[..., Any],
    span_name: str,
    attributes: AttributeSpec,
    func: Callable[..., Any],
    *,
    input_resolver: Any = None,
    variables_resolver: Any = None,
    evaluators: Any = None,
    post_evaluators: Any = None,
    evaluator_config: Any = None,
    evaluator_metadata: Any = None,
    output_resolver: Callable[[Any], Any] | None = None,
    apply_pre: Callable[[Any, Any], None] | None = None,
    apply_post: Callable[[Any, Any], None] | None = None,
) -> Callable[..., Any]:
    """
    Higher-order utility to wrap a function with span instrumentation.
    """
    is_async = inspect.iscoroutinefunction(func)

    if is_async:

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            computed_attrs = _resolve_attributes(attributes, args, kwargs)
            bound = _resolve_bound_arguments(func, args, kwargs)
            input_payload = _resolve_payload_from_bound(input_resolver, bound)
            variables_payload = _resolve_variables_payload(variables_resolver, bound)
            pre_evaluators = _resolve_evaluators_payload(evaluators, bound)

            with span_factory(
                span_name,
                attributes=computed_attrs,
                input_payload=input_payload,
                variables=variables_payload,
                evaluators=pre_evaluators,
            ) as span:
                # Set evaluator config and metadata if provided
                if evaluator_config is not None:
                    resolved_config = evaluator_config(bound) if callable(evaluator_config) else evaluator_config
                    if resolved_config is not None:
                        span.set_evaluator_config(resolved_config)
                if evaluator_metadata is not None:
                    resolved_metadata = evaluator_metadata(bound) if callable(evaluator_metadata) else evaluator_metadata
                    if resolved_metadata is not None:
                        span.set_evaluator_metadata(resolved_metadata)

                if apply_pre:
                    apply_pre(span, bound)
                try:
                    result = await func(*args, **kwargs)
                    transformed = output_resolver(result) if output_resolver else result
                    span.set_output(transformed)
                    if apply_post:
                        apply_post(span, result)
                    post_specs = _resolve_evaluators_payload(post_evaluators, bound, result)
                    if post_specs:
                        span.add_evaluators(*post_specs)
                    span.set_status(StatusCode.OK)
                    return result
                except Exception as exc:  # pragma: no cover - passthrough
                    span.record_exception(exc)
                    span.set_output({"error": str(exc)})
                    span.set_status(StatusCode.ERROR, str(exc))
                    raise

        return async_wrapper

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        computed_attrs = _resolve_attributes(attributes, args, kwargs)
        bound = _resolve_bound_arguments(func, args, kwargs)
        input_payload = _resolve_payload_from_bound(input_resolver, bound)
        variables_payload = _resolve_variables_payload(variables_resolver, bound)
        pre_evaluators = _resolve_evaluators_payload(evaluators, bound)

        with span_factory(
            span_name,
            attributes=computed_attrs,
            input_payload=input_payload,
            variables=variables_payload,
            evaluators=pre_evaluators,
        ) as span:
            # Set evaluator config and metadata if provided
            if evaluator_config is not None:
                resolved_config = evaluator_config(bound) if callable(evaluator_config) else evaluator_config
                if resolved_config is not None:
                    span.set_evaluator_config(resolved_config)
            if evaluator_metadata is not None:
                resolved_metadata = evaluator_metadata(bound) if callable(evaluator_metadata) else evaluator_metadata
                if resolved_metadata is not None:
                    span.set_evaluator_metadata(resolved_metadata)

            if apply_pre:
                apply_pre(span, bound)
            try:
                result = func(*args, **kwargs)
                transformed = output_resolver(result) if output_resolver else result
                span.set_output(transformed)
                if apply_post:
                    apply_post(span, result)
                post_specs = _resolve_evaluators_payload(post_evaluators, bound, result)
                if post_specs:
                    span.add_evaluators(*post_specs)
                span.set_status(StatusCode.OK)
                return result
            except Exception as exc:  # pragma: no cover - passthrough
                span.record_exception(exc)
                span.set_output({"error": str(exc)})
                span.set_status(StatusCode.ERROR, str(exc))
                raise

    return sync_wrapper


def trace_span(
    name: str | None = None,
    *,
    attributes: AttributeSpec = None,
    input: Any = None,
    output: Callable[[Any], Any] | None = None,
    variables: Any = None,
    evaluators: Any = None,
    post_evaluators: Any = None,
    evaluator_config: Any = None,
    evaluator_metadata: Any = None,
) -> Callable[[F], F]:
    """Decorator for general-purpose spans."""

    def decorator(func: F) -> F:
        span_name = name or f"{func.__module__}.{func.__qualname__}"
        return _wrap_with_span(
            trace_span_cm,
            span_name,
            attributes,
            func,
            input_resolver=input,
            variables_resolver=variables,
            evaluators=evaluators,
            post_evaluators=post_evaluators,
            evaluator_config=evaluator_config,
            evaluator_metadata=evaluator_metadata,
            output_resolver=output,
        )  # type: ignore[return-value]

    return decorator


def trace_operation(
    name: str | None = None,
    *,
    attributes: AttributeSpec = None,
    capture_io: bool = False,
) -> Callable[[F], F]:
    """Deprecated alias for ``trace_span``."""
    warnings.warn(
        "trace_operation is deprecated; use trace_span instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return trace_span(name=name, attributes=attributes)


def _extract_first(bound, keys: tuple[str, ...]) -> Any | None:
    if not bound:
        return None
    for key in keys:
        if key in bound.arguments:
            return bound.arguments[key]
    return None


def _default_generation_input(bound: inspect.BoundArguments | None) -> Any:
    value = _extract_first(bound, ("prompt", "input", "inputs", "messages", "question"))
    if value is not None:
        return value
    return dict(bound.arguments) if bound else None


def _default_generation_variables(bound: inspect.BoundArguments | None) -> Mapping[str, Any] | None:
    value = _extract_first(bound, ("variables", "params", "context"))
    return value if isinstance(value, Mapping) else None


def _default_retrieval_input(bound: inspect.BoundArguments | None) -> Any:
    value = _extract_first(bound, ("query", "question", "text", "search"))
    if value is not None:
        return value
    return dict(bound.arguments) if bound else None


def _default_retrieval_variables(bound: inspect.BoundArguments | None) -> Mapping[str, Any] | None:
    value = _extract_first(bound, ("filters", "metadata", "options"))
    return value if isinstance(value, Mapping) else None


def _default_tool_input(bound: inspect.BoundArguments | None) -> Any:
    value = _extract_first(bound, ("tool_input", "payload", "input", "data"))
    if value is not None:
        return value
    return dict(bound.arguments) if bound else None


def _default_event_input(bound: inspect.BoundArguments | None) -> Any:
    value = _extract_first(bound, ("payload", "event", "data"))
    if value is not None:
        return value
    return dict(bound.arguments) if bound else None


def _default_function_variables(bound: inspect.BoundArguments | None) -> Mapping[str, Any] | None:
    value = _extract_first(bound, ("variables", "metadata", "context"))
    return value if isinstance(value, Mapping) else None


def _serialize_prompt(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except Exception:
        return str(value)


def _extract_completion(result: Any) -> str | None:
    if result is None:
        return None
    if isinstance(result, str):
        return result

    data: dict[str, Any] | None = None
    if isinstance(result, dict):
        data = result
    elif hasattr(result, "model_dump"):
        try:
            data = result.model_dump()
        except Exception:
            data = None
    elif hasattr(result, "dict"):
        try:
            data = result.dict()
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


def _extract_usage(result: Any) -> tuple[int | None, int | None]:
    usage_section: Any | None = None
    if isinstance(result, dict):
        usage_section = result.get("usage")
    elif hasattr(result, "usage"):
        usage_section = result.usage
    elif hasattr(result, "model_dump"):
        try:
            dumped = result.model_dump()
            usage_section = dumped.get("usage")
        except Exception:
            usage_section = None
    if not isinstance(usage_section, dict):
        return None, None
    input_tokens = usage_section.get("prompt_tokens") or usage_section.get("input_tokens")
    output_tokens = usage_section.get("completion_tokens") or usage_section.get("output_tokens")
    input_tokens = int(input_tokens) if isinstance(input_tokens, (int, float)) else None
    output_tokens = int(output_tokens) if isinstance(output_tokens, (int, float)) else None
    return input_tokens, output_tokens


def _apply_llm_request_metadata(span: LLMSpanHandle, bound) -> None:
    if not bound:
        return
    model = _extract_first(bound, ("model", "model_name"))
    if isinstance(model, str):
        span.set_model(model)
    prompt = _extract_first(bound, ("prompt", "input", "inputs", "messages", "question"))
    serialized = _serialize_prompt(prompt)
    if serialized and trace_content_enabled():
        span.set_prompt(serialized)


def _apply_llm_response_metadata(span: LLMSpanHandle, result: Any) -> None:
    completion = _extract_completion(result)
    if completion and trace_content_enabled():
        span.set_completion(completion)
    input_tokens, output_tokens = _extract_usage(result)
    if input_tokens is not None or output_tokens is not None:
        span.set_tokens(input=input_tokens, output=output_tokens)


def trace_generation(
    name: str | None = None,
    *,
    attributes: AttributeSpec = None,
    input: Any = None,
    output: Callable[[Any], Any] | None = None,
    variables: Any = None,
    evaluators: Any = None,
    post_evaluators: Any = None,
    evaluator_config: Any = None,
    evaluator_metadata: Any = None,
) -> Callable[[F], F]:
    """Decorator specialized for LLM generation spans.

    Args:
        name: Optional span name. Defaults to module.function_name.
        attributes: Static or callable attributes to add to the span.
        input: Resolver for input payload (string, sequence, callable, or None).
        output: Callable to transform the return value before recording.
        variables: Resolver for span variables.
        evaluators: Evaluator specifications to attach before execution.
        post_evaluators: Evaluator specifications to attach after execution.
        evaluator_config: Config for evaluators (EvaluatorConfig, dict, or callable).
        evaluator_metadata: Metadata for evaluators (dict or callable returning dict).
    """

    def decorator(func: F) -> F:
        span_name = name or f"{func.__module__}.{func.__qualname__}"

        def apply_pre(span, bound):
            _apply_llm_request_metadata(span, bound)

        def apply_post(span, result):
            _apply_llm_response_metadata(span, result)

        input_resolver = input if input is not None else _default_generation_input
        variables_resolver = variables if variables is not None else _default_generation_variables

        return _wrap_with_span(
            trace_generation_cm,
            span_name,
            attributes,
            func,
            input_resolver=input_resolver,
            variables_resolver=variables_resolver,
            evaluators=evaluators,
            post_evaluators=post_evaluators,
            evaluator_config=evaluator_config,
            evaluator_metadata=evaluator_metadata,
            output_resolver=output,
            apply_pre=apply_pre,
            apply_post=apply_post,
        )  # type: ignore[return-value]

    return decorator


def trace_llm(
    name: str | None = None,
    *,
    attributes: AttributeSpec = None,
    input: Any = None,
    output: Callable[[Any], Any] | None = None,
    variables: Any = None,
    evaluators: Any = None,
    post_evaluators: Any = None,
) -> Callable[[F], F]:
    """Deprecated alias for :func:`trace_generation`."""
    warnings.warn(
        "trace_llm is deprecated; use trace_generation instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return trace_generation(
        name=name,
        attributes=attributes,
        input=input,
        output=output,
        variables=variables,
        evaluators=evaluators,
        post_evaluators=post_evaluators,
    )


def trace_retrieval(
    name: str | None = None,
    *,
    attributes: AttributeSpec = None,
    input: Any = None,
    output: Callable[[Any], Any] | None = None,
    variables: Any = None,
    evaluators: Any = None,
    post_evaluators: Any = None,
) -> Callable[[F], F]:
    """Decorator for retrieval/vector search spans."""

    def decorator(func: F) -> F:
        span_name = name or f"{func.__module__}.{func.__qualname__}"
        input_resolver = input if input is not None else _default_retrieval_input
        variables_resolver = variables if variables is not None else _default_retrieval_variables

        def apply_pre(span, bound):
            query = _resolve_payload_from_bound(input_resolver, bound)
            if isinstance(query, str):
                span.set_query(query)

        return _wrap_with_span(
            trace_retrieval_cm,
            span_name,
            attributes,
            func,
            input_resolver=input_resolver,
            variables_resolver=variables_resolver,
            evaluators=evaluators,
            post_evaluators=post_evaluators,
            output_resolver=output,
            apply_pre=apply_pre,
        )  # type: ignore[return-value]

    return decorator


def trace_function(
    name: str | None = None,
    *,
    attributes: AttributeSpec = None,
    input: Any = None,
    output: Callable[[Any], Any] | None = None,
    variables: Any = None,
    evaluators: Any = None,
    post_evaluators: Any = None,
    function_name: str | Callable[[inspect.BoundArguments | None], str] | None = None,
    stage: str | Callable[[inspect.BoundArguments | None], str] | None = None,
) -> Callable[[F], F]:
    """Decorator for compute/function spans."""

    def decorator(func: F) -> F:
        span_name = name or f"{func.__module__}.{func.__qualname__}"
        variables_resolver = variables if variables is not None else _default_function_variables

        def resolve_callable(value, bound):
            if value is None:
                return None
            if callable(value):
                return value(bound)
            return value

        def apply_pre(span, bound):
            resolved_name = resolve_callable(function_name, bound) or span_name
            span.set_function_name(resolved_name)
            resolved_stage = resolve_callable(stage, bound)
            if resolved_stage:
                span.set_stage(resolved_stage)

        return _wrap_with_span(
            trace_function_cm,
            span_name,
            attributes,
            func,
            input_resolver=input,
            variables_resolver=variables_resolver,
            evaluators=evaluators,
            post_evaluators=post_evaluators,
            output_resolver=output,
            apply_pre=apply_pre,
        )  # type: ignore[return-value]

    return decorator


def trace_tool(
    name: str | None = None,
    *,
    attributes: AttributeSpec = None,
    input: Any = None,
    output: Callable[[Any], Any] | None = None,
    variables: Any = None,
    evaluators: Any = None,
    post_evaluators: Any = None,
    tool_name: str | Callable[[inspect.BoundArguments | None], str] | None = None,
) -> Callable[[F], F]:
    """Decorator for tool invocation spans."""

    def decorator(func: F) -> F:
        span_name = name or f"{func.__module__}.{func.__qualname__}"
        input_resolver = input if input is not None else _default_tool_input

        def apply_pre(span, bound):
            value = tool_name(bound) if callable(tool_name) else tool_name
            if value:
                span.set_tool_name(value)

        return _wrap_with_span(
            trace_tool_cm,
            span_name,
            attributes,
            func,
            input_resolver=input_resolver,
            variables_resolver=variables,
            evaluators=evaluators,
            post_evaluators=post_evaluators,
            output_resolver=output,
            apply_pre=apply_pre,
        )  # type: ignore[return-value]

    return decorator


def trace_event(
    name: str | None = None,
    *,
    attributes: AttributeSpec = None,
    input: Any = None,
    output: Callable[[Any], Any] | None = None,
    variables: Any = None,
    evaluators: Any = None,
    post_evaluators: Any = None,
    event_type: str | Callable[[inspect.BoundArguments | None], str] | None = None,
) -> Callable[[F], F]:
    """Decorator for custom event spans."""

    def decorator(func: F) -> F:
        span_name = name or f"{func.__module__}.{func.__qualname__}"
        input_resolver = input if input is not None else _default_event_input

        def apply_pre(span, bound):
            payload = _resolve_payload_from_bound(input_resolver, bound)
            if payload is not None:
                span.set_payload(payload)
            value = event_type(bound) if callable(event_type) else event_type
            if value:
                span.set_event_type(value)

        return _wrap_with_span(
            trace_event_cm,
            span_name,
            attributes,
            func,
            input_resolver=input_resolver,
            variables_resolver=variables,
            evaluators=evaluators,
            post_evaluators=post_evaluators,
            output_resolver=output,
            apply_pre=apply_pre,
        )  # type: ignore[return-value]

    return decorator


def evaluator(
    slugs: str | Sequence[str],
    *,
    sample_rate: float = 1.0,
    metadata: Mapping[str, Any] | Callable[..., Mapping[str, Any]] | None = None,
) -> Callable[[F], F]:
    """
    Decorator that propagates evaluator slugs through OpenTelemetry context.

    The decorator uses :func:`with_evaluators` to push evaluator identifiers into
    the active context. Any span created while the decorated function executes—
    whether by automatic instrumentation or manual tracing—receives the slugs via
    :class:`BasaltCallEvaluatorProcessor`. Manual spans also receive the slugs
    immediately through :class:`~basalt.observability.context_managers.SpanHandle`.

    Args:
        slugs: One or more evaluator slugs to attach.
        sample_rate: Probability (0.0-1.0) that this evaluator will be attached to spans.
                    Default is 1.0 (100% of traces). Use lower values to reduce evaluation costs.
        metadata: Optional metadata for evaluators. Can be a static dict or a callable that
                 receives function arguments and returns a dict.

    Example - Basic usage:
        >>> @evaluator("joke-quality")
        ... @trace_generation(name="gemini.summarize_joke")
        ... def summarize_joke_with_gemini(joke: str) -> str:
        ...     return call_llm(joke)

    Example - With sample rate (50% of traces):
        >>> @evaluator(["hallucination-check"], sample_rate=0.5)
        ... @trace_generation(name="my.llm.call")
        ... def call_llm(prompt: str) -> str:
        ...     return sdk.generate(prompt)

    Example - With metadata callable:
        >>> @evaluator(
        ...     "prompt-vars-check",
        ...     sample_rate=0.5,
        ...     metadata=lambda user_query, context_vector_count=10, **kwargs: {
        ...         "user_input": user_query,
        ...         "model_context": context_vector_count
        ...     }
        ... )
        ... @trace_generation(name="my.llm.call")
        ... def call_llm(user_query: str, context_vector_count: int = 10) -> str:
        ...     return sdk.generate(user_query)
    """

    if isinstance(slugs, str):
        slug_list = [slugs.strip()]
    elif isinstance(slugs, Sequence):
        slug_list = [str(slug).strip() for slug in slugs if str(slug).strip()]
    else:
        raise TypeError("Evaluator slugs must be provided as a string or sequence of strings.")

    slug_list = list(dict.fromkeys(slug_list))
    if not slug_list:
        raise ValueError("At least one evaluator slug must be provided.")
    if not 0.0 <= sample_rate <= 1.0:
        raise ValueError("sample_rate must be within [0.0, 1.0].")

    def decorator(func: F) -> F:
        is_async = inspect.iscoroutinefunction(func)

        def _should_attach() -> bool:
            if sample_rate >= 1.0:
                return True
            return random.random() <= sample_rate

        def _resolve_metadata(args, kwargs):
            """Resolve metadata from callable or static value."""
            if metadata is None:
                return None
            if callable(metadata):
                try:
                    resolved = metadata(*args, **kwargs)
                    if resolved and isinstance(resolved, Mapping):
                        return resolved
                except Exception:
                    pass  # Silently skip if metadata resolution fails
                return None
            elif isinstance(metadata, Mapping):
                return metadata
            return None

        if is_async:

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not _should_attach():
                    return await func(*args, **kwargs)

                from .context_managers import EvaluatorConfig

                # Resolve metadata before entering context
                resolved_metadata = _resolve_metadata(args, kwargs)
                config = EvaluatorConfig(sample_rate=sample_rate)

                with with_evaluators(slug_list, config=config, metadata=resolved_metadata):
                    return await func(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not _should_attach():
                return func(*args, **kwargs)

            from .context_managers import EvaluatorConfig

            # Resolve metadata before entering context
            resolved_metadata = _resolve_metadata(args, kwargs)
            config = EvaluatorConfig(sample_rate=sample_rate)

            with with_evaluators(slug_list, config=config, metadata=resolved_metadata):
                return func(*args, **kwargs)

        return sync_wrapper  # type: ignore[return-value]

    return decorator
