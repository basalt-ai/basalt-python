from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Sequence
from contextlib import ContextDecorator
from typing import TYPE_CHECKING, Any, TypeVar

from opentelemetry.trace import StatusCode

if TYPE_CHECKING:
    from basalt.experiments.models import Experiment
    from basalt.prompts.models import Prompt

from .context_managers import (
    ROOT_SPAN_CONTEXT_KEY,
    EvaluationConfig,
    EventSpanHandle,
    FunctionSpanHandle,
    LLMSpanHandle,
    RetrievalSpanHandle,
    SpanHandle,
    StartSpanHandle,
    ToolSpanHandle,
    _async_with_span_handle,
    _set_trace_organization,
    _set_trace_user,
    _with_span_handle,
    get_current_otel_span,
    get_current_span_handle,
    get_root_span_handle,
)
from .decorators import ObserveKind
from .types import Identity
from .utils import (
    apply_llm_request_metadata,
    apply_llm_response_metadata,
    default_generation_input,
    default_generation_variables,
    default_retrieval_input,
    default_retrieval_variables,
    resolve_attributes,
    resolve_bound_arguments,
    resolve_evaluators_payload,
    resolve_identity_payload,
    resolve_payload_from_bound,
    resolve_variables_payload,
)

F = TypeVar("F", bound=Callable[..., Any])


class StartObserve(ContextDecorator):
    """
    Entry point for Basalt observability.
    Must be used as the root span of a trace.
    """

    def __init__(
        self,
        feature_slug: str,
        name: str,
        *,
        identity: Identity | Callable[..., Identity | None] | None = None,
        evaluate_config: EvaluationConfig | None = None,
        evaluators: Sequence[Any] | None = None,
        experiment: str | Experiment | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        # Validate feature_slug is provided and non-empty
        if not feature_slug or not isinstance(feature_slug, str) or not feature_slug.strip():
            raise ValueError(
                "feature_slug is required and must be a non-empty string. "
                "Please provide a valid feature identifier (e.g., 'user_authentication', 'payment_processing')."
            )

        # Validate name is provided and non-empty
        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError(
                "name is required and must be a non-empty string. "
                "Please provide a descriptive name for this span (e.g., 'user_authentication_flow', 'process_payment')."
            )

        self.name = name.strip()
        self.feature_slug = feature_slug.strip()
        self.identity_resolver = identity
        self.evaluate_config = evaluate_config
        self.evaluators = evaluators
        self.experiment = experiment
        self._metadata = metadata
        self._span_handle: StartSpanHandle | None = None
        self._ctx_manager = None

    def __enter__(self) -> StartSpanHandle:
        span_name = self.name

        # Resolve identity
        user_identity, org_identity = resolve_identity_payload(self.identity_resolver, None)

        # Initialize context manager
        self._ctx_manager = _with_span_handle(
            name=span_name,
            attributes=None,
            tracer_name="basalt.observability",
            handle_cls=StartSpanHandle,
            span_type="basalt_trace",
            user=user_identity,
            organization=org_identity,
            evaluators=self.evaluators,
            feature_slug=self.feature_slug,
            metadata=self._metadata,
        )
        span = self._ctx_manager.__enter__()
        # Type assertion: we know this is StartSpanHandle since we passed it as handle_cls
        assert isinstance(span, StartSpanHandle)
        self._span_handle = span

        # Set evaluation config if provided
        if self.evaluate_config is not None:
            self._span_handle.set_evaluation_config(self.evaluate_config)

        # Attach experiment if provided (only on root span, which this is)
        if self.experiment:
            self._apply_experiment(self._span_handle)

        return self._span_handle

    def __exit__(self, exc_type, exc_value, traceback):
        if self._ctx_manager:
            return self._ctx_manager.__exit__(exc_type, exc_value, traceback)
        return None

    def __call__(self, func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Resolve identity from args/kwargs if needed
            bound = resolve_bound_arguments(func, args, kwargs)
            user_identity, org_identity = resolve_identity_payload(self.identity_resolver, bound)
            pre_evaluators = resolve_evaluators_payload(self.evaluators, bound)

            span_name = self.name
            with _with_span_handle(
                name=span_name,
                attributes=None,
                tracer_name="basalt.observability",
                handle_cls=StartSpanHandle,
                span_type="basalt_trace",
                user=user_identity,
                organization=org_identity,
                evaluators=pre_evaluators,
                feature_slug=self.feature_slug,
                metadata=self._metadata,
            ) as handle:
                # Type assertion: we know this is StartSpanHandle since we passed it as handle_cls
                assert isinstance(handle, StartSpanHandle)
                span = handle

                # Set evaluation config if provided
                if self.evaluate_config is not None:
                    span.set_evaluation_config(self.evaluate_config)

                # Attach experiment if provided
                if self.experiment:
                    self._apply_experiment(span)

                try:
                    return func(*args, **kwargs)
                except Exception:
                    raise

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                bound = resolve_bound_arguments(func, args, kwargs)
                user_identity, org_identity = resolve_identity_payload(self.identity_resolver, bound)
                pre_evaluators = resolve_evaluators_payload(self.evaluators, bound)

                span_name = self.name
                with _with_span_handle(
                    name=span_name,
                    attributes=None,
                    tracer_name="basalt.observability",
                    handle_cls=StartSpanHandle,
                    span_type="basalt_trace",
                    user=user_identity,
                    organization=org_identity,
                    evaluators=pre_evaluators,
                    feature_slug=self.feature_slug,
                    metadata=self._metadata,
                ) as handle:
                    # Type assertion: we know this is StartSpanHandle since we passed it as handle_cls
                    assert isinstance(handle, StartSpanHandle)
                    span = handle

                    # Set evaluation config if provided
                    if self.evaluate_config is not None:
                        span.set_evaluation_config(self.evaluate_config)

                    if self.experiment:
                        self._apply_experiment(span)
                    try:
                        return await func(*args, **kwargs)
                    except Exception:
                        raise

            return async_wrapper  # type: ignore

        return wrapper  # type: ignore

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _apply_experiment(self, span: StartSpanHandle | None) -> None:
        """Apply experiment metadata to the provided span.

        Supports either a string experiment ID or an
        `Experiment` dataclass instance from `basalt.experiments.models`.
        """
        if span is None or not self.experiment:
            return

        exp_id: str | None = None

        # Handle string ID case
        if isinstance(self.experiment, str):
            exp_id = self.experiment
        # Check if it's an Experiment dataclass by looking for the 'id' attribute
        # (avoid isinstance to prevent circular import at runtime)
        elif hasattr(self.experiment, "id"):
            exp_id = self.experiment.id  # type: ignore[attr-defined]

        if not exp_id:
            return  # Must have an id to attach

        span.set_experiment(
            experiment=exp_id,
        )


class Observe(ContextDecorator):
    """
    Unified observability interface for Basalt.
    Acts as both a decorator and a context manager.
    """

    def __init__(
        self,
        name: str,
        kind: ObserveKind | str = ObserveKind.SPAN,
        *,
        metadata: dict[str, Any] | None = None,
        evaluators: Sequence[Any] | None = None,
        input: Any = None,
        output: Any = None,
        variables: dict[str, Any] | None = None,
        prompt: Prompt | None = None,
    ):
        # Validate name is provided and non-empty
        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError(
                "name is required and must be a non-empty string. "
                "Please provide a descriptive name for this span (e.g., 'llm_generation', 'vector_search')."
            )

        self.name = name.strip()
        self.kind = kind
        self._metadata = metadata
        self.evaluators = evaluators
        self.input_resolver = input
        self.output_resolver = output
        self.variables_resolver = variables
        self.prompt = prompt
        self._span_handle: SpanHandle | None = None
        self._ctx_manager = None

    @staticmethod
    def _get_config_for_kind(kind_str: str):
        """Return handle class, tracer name, and default resolvers for the kind."""
        if kind_str == "generation":
            return (
                LLMSpanHandle,
                "basalt.observability.generation",
                default_generation_input,
                default_generation_variables,
            )
        elif kind_str == "retrieval":
            return (
                RetrievalSpanHandle,
                "basalt.observability.retrieval",
                default_retrieval_input,
                default_retrieval_variables,
            )
        elif kind_str == "tool":
            return (
                ToolSpanHandle,
                "basalt.observability.tool",
                None,
                None,
            )
        elif kind_str == "function":
            return (
                FunctionSpanHandle,
                "basalt.observability.function",
                None,
                None,
            )
        elif kind_str == "event":
            return (
                EventSpanHandle,
                "basalt.observability.event",
                None,
                None,
            )
        else:
            return (
                SpanHandle,
                "basalt.observability",
                None,
                None,
            )

    def __enter__(self) -> SpanHandle:
        span_name = self.name

        if isinstance(self.kind, ObserveKind):
            kind_str = self.kind.value
        else:
            kind_str = str(self.kind).lower()
            # Validate that the string kind is valid
            valid_kinds = {k.value for k in ObserveKind}
            if kind_str not in valid_kinds:
                raise ValueError(f"Invalid kind '{kind_str}'. Must be one of: {', '.join(sorted(valid_kinds))}")

        # Reject ROOT kind
        if kind_str == ObserveKind.ROOT.value:
            raise ValueError(
                f"Cannot use kind='{ObserveKind.ROOT.value}' with Observe. "
                f"Use StartObserve (start_observe) for root spans."
            )

        handle_cls, tracer_name, _, _ = self._get_config_for_kind(kind_str)

        # Process prompt parameter if provided
        prompt_attrs = None
        if self.prompt is not None:
            # Prepare prompt attributes for span
            import json

            prompt_attrs = {
                "basalt.prompt.slug": self.prompt.slug,
                "basalt.prompt.version": self.prompt.version,
                "basalt.prompt.model.provider": self.prompt.model.provider,
                "basalt.prompt.model.model": self.prompt.model.model,
            }

            # Store prompt.variables separately if available (must serialize to JSON for OpenTelemetry)
            if self.prompt.variables:
                prompt_attrs["basalt.prompt.variables"] = json.dumps(self.prompt.variables)

        # Check for root span
        from opentelemetry import context as otel_context

        if not otel_context.get_value(ROOT_SPAN_CONTEXT_KEY):
            import logging

            logger = logging.getLogger(__name__)
            logger.warning("Observe used without a preceding start_observe. This may lead to missing trace context.")

        self._ctx_manager = _with_span_handle(
            name=span_name,
            attributes=prompt_attrs,
            tracer_name=tracer_name,
            handle_cls=handle_cls,
            span_type=kind_str,
            evaluators=self.evaluators,
            metadata=self._metadata,
            # In context manager mode, we don't auto-resolve input/vars from args
            # User must call observe.input() or pass explicit input_payload if we added it to __init__
            # But __init__ has resolvers, not values.
            # So we rely on manual calls or future extensions.
        )
        self._span_handle = self._ctx_manager.__enter__()
        return self._span_handle

    def __exit__(self, exc_type, exc_value, traceback):
        if self._ctx_manager:
            return self._ctx_manager.__exit__(exc_type, exc_value, traceback)
        return None

    def __call__(self, func: F) -> F:
        if isinstance(self.kind, ObserveKind):
            kind_str = self.kind.value
        else:
            kind_str = str(self.kind).lower()

        # Reject ROOT kind
        if kind_str == ObserveKind.ROOT.value:
            raise ValueError(
                f"Cannot use kind='{ObserveKind.ROOT.value}' with Observe. "
                f"Use StartObserve (start_observe) for root spans."
            )

        handle_cls, tracer_name, default_input, default_vars = self._get_config_for_kind(kind_str)

        # Use defaults if not provided
        input_resolver = self.input_resolver if self.input_resolver is not None else default_input
        variables_resolver = self.variables_resolver if self.variables_resolver is not None else default_vars

        # Process prompt parameter if provided
        prompt_metadata = {}
        if self.prompt is not None:
            import json

            # Override input resolver with prompt.text
            def input_resolver(bound):
                return self.prompt.text

            # Prepare prompt metadata for span attributes
            prompt_metadata = {
                "basalt.prompt.slug": self.prompt.slug,
                "basalt.prompt.version": self.prompt.version,
                "basalt.prompt.model.provider": self.prompt.model.provider,
                "basalt.prompt.model.model": self.prompt.model.model,
            }

            # Store prompt.variables separately if available (must serialize to JSON for OpenTelemetry)
            if self.prompt.variables:
                prompt_metadata["basalt.prompt.variables"] = json.dumps(self.prompt.variables)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            computed_metadata = resolve_attributes(self._metadata, args, kwargs)
            bound = resolve_bound_arguments(func, args, kwargs)
            input_payload = resolve_payload_from_bound(input_resolver, bound)
            variables_payload = resolve_variables_payload(variables_resolver, bound)
            pre_evaluators = resolve_evaluators_payload(self.evaluators, bound)

            # Check for root span
            from opentelemetry import context as otel_context

            if not otel_context.get_value(ROOT_SPAN_CONTEXT_KEY):
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    "Observe used without a preceding start_observe. This may lead to missing trace context."
                )

            # Pre-hooks
            def apply_pre(span, bound):
                if kind_str == "generation" and isinstance(span, LLMSpanHandle):
                    apply_llm_request_metadata(span, bound)
                elif kind_str == "retrieval" and isinstance(span, RetrievalSpanHandle):
                    query = resolve_payload_from_bound(input_resolver, bound)
                    if isinstance(query, str):
                        span.set_query(query)

            # Post-hooks
            def apply_post(span, result):
                if kind_str == "generation" and isinstance(span, LLMSpanHandle):
                    apply_llm_response_metadata(span, result)

            with _with_span_handle(
                name=self.name,
                attributes=prompt_metadata if prompt_metadata else None,
                tracer_name=tracer_name,
                handle_cls=handle_cls,
                span_type=kind_str,
                input_payload=input_payload,
                variables=variables_payload,
                evaluators=pre_evaluators,
                metadata=computed_metadata,
            ) as span:
                if apply_pre:
                    apply_pre(span, bound)

                try:
                    result = func(*args, **kwargs)

                    transformed = self.output_resolver(result) if self.output_resolver else result
                    span.set_output(transformed)

                    if apply_post:
                        apply_post(span, result)

                    return result
                except Exception:
                    span.set_output({"error": "Exception occurred"})
                    raise

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                computed_metadata = resolve_attributes(self._metadata, args, kwargs)
                bound = resolve_bound_arguments(func, args, kwargs)
                input_payload = resolve_payload_from_bound(input_resolver, bound)
                variables_payload = resolve_variables_payload(variables_resolver, bound)
                pre_evaluators = resolve_evaluators_payload(self.evaluators, bound)

                # Check for root span
                from opentelemetry import context as otel_context

                if not otel_context.get_value(ROOT_SPAN_CONTEXT_KEY):
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.warning(
                        "Observe used without a preceding start_observe. This may lead to missing trace context."
                    )

                # Pre-hooks (same as sync)
                def apply_pre(span, bound):
                    if kind_str == "generation" and isinstance(span, LLMSpanHandle):
                        apply_llm_request_metadata(span, bound)
                    elif kind_str == "retrieval" and isinstance(span, RetrievalSpanHandle):
                        query = resolve_payload_from_bound(input_resolver, bound)
                        if isinstance(query, str):
                            span.set_query(query)

                # Post-hooks (same as sync)
                def apply_post(span, result):
                    if kind_str == "generation" and isinstance(span, LLMSpanHandle):
                        apply_llm_response_metadata(span, result)

                with _with_span_handle(
                    name=self.name,
                    attributes=prompt_metadata if prompt_metadata else None,
                    tracer_name=tracer_name,
                    handle_cls=handle_cls,
                    span_type=kind_str,
                    input_payload=input_payload,
                    variables=variables_payload,
                    evaluators=pre_evaluators,
                    metadata=computed_metadata,
                ) as span:
                    if apply_pre:
                        apply_pre(span, bound)

                    try:
                        result = await func(*args, **kwargs)

                        transformed = self.output_resolver(result) if self.output_resolver else result
                        span.set_output(transformed)

                        if apply_post:
                            apply_post(span, result)

                        return result
                    except Exception:
                        span.set_output({"error": "Exception occurred"})
                        raise

            return async_wrapper  # type: ignore

        return wrapper  # type: ignore

    # Static Domain Methods

    @staticmethod
    def _identify(user: str | dict[str, Any] | None = None, organization: str | dict[str, Any] | None = None) -> None:
        """Set the user and/or organization identity for the current context."""
        if user:
            if isinstance(user, str):
                _set_trace_user(user_id=user)
            elif isinstance(user, dict):
                _set_trace_user(user_id=user.get("id", "unknown"), name=user.get("name"))

        if organization:
            if isinstance(organization, str):
                _set_trace_organization(organization_id=organization)
            elif isinstance(organization, dict):
                _set_trace_organization(organization_id=organization.get("id", "unknown"), name=organization.get("name"))

    @staticmethod
    def _root_span() -> StartSpanHandle | None:
        """Get the root span handle of the current trace.

        Returns the handle for the root span, enabling late-binding of
        identify() or metadata from deeply nested contexts.

        Returns:
            StartSpanHandle if a root span exists, None otherwise.
        """
        return get_root_span_handle()

    @staticmethod
    def set_input(data: str | dict[str, Any]) -> None:
        """Set input data for the current span."""
        handle = get_current_span_handle()
        if handle:
            handle.set_input(data)

    @staticmethod
    def set_output(data: str | dict[str, Any]) -> None:
        """Set output data for the current span."""
        handle = get_current_span_handle()
        if handle:
            handle.set_output(data)

    @staticmethod
    def evaluate(evaluator: Sequence[str]) -> None:
        """Attach an evaluator to the current span."""
        handle = get_current_span_handle()
        if handle:
            handle.add_evaluators(evaluator)

    @staticmethod
    def _status(status: StatusCode | str, message: str | None = None) -> None:
        """Set the status of the current span."""
        span = get_current_otel_span()
        if not span:
            return

        if isinstance(status, str):
            status_map = {"ok": StatusCode.OK, "error": StatusCode.ERROR, "unset": StatusCode.UNSET}
            code = status_map.get(status.lower(), StatusCode.UNSET)
        else:
            code = status

        span.set_status(code, description=message)

    @staticmethod
    def _fail(exception: BaseException | str) -> None:
        """Record an error/exception and set status to ERROR."""
        span = get_current_otel_span()
        if not span:
            return

        if isinstance(exception, BaseException):
            span.record_exception(exception)
            span.set_status(StatusCode.ERROR, str(exception))
        else:
            span.set_status(StatusCode.ERROR, str(exception))

    @staticmethod
    def set_prompt(prompt: Prompt) -> None:
        """Set prompt metadata on the current span."""
        handle = get_current_span_handle()
        if handle:
            handle.set_prompt(prompt)

    @staticmethod
    def set_metadata(data: dict[str, Any]) -> None:
        """Merge metadata into the current span as a JSON object at basalt.metadata.

        Existing metadata keys are preserved unless overridden by ``data``.
        """
        handle = get_current_span_handle()
        if handle:
            handle.set_metadata(data)

    @staticmethod
    def set_attributes(attributes: dict[str, Any]) -> None:
        """Set multiple attributes on the current span."""
        handle = get_current_span_handle()
        if handle:
            handle.set_attributes(attributes)

    @staticmethod
    def set_io(
        *,
        input_payload: Any | None = None,
        output_payload: Any | None = None,
        variables: dict[str, Any] | None = None,
    ) -> None:
        """Set input, output, and variables for the current span."""
        handle = get_current_span_handle()
        if handle:
            handle.set_io(input_payload=input_payload, output_payload=output_payload, variables=variables)

    @staticmethod
    def inject_for_auto_instrumentation(
        *,
        input_payload: Any | None = None,
        output_payload: Any | None = None,
        prompt: Prompt | None = None,
        metadata: dict[str, Any] | None = None,
        variables: dict[str, Any] | None = None,
    ) -> None:
        """
        Inject data into the next auto-instrumented span.

        This method stores data in OpenTelemetry context that will be applied
        to the next span created by auto-instrumentation libraries (OpenAI,
        Anthropic, LangChain, ChromaDB, etc.). The data is automatically cleared
        after being applied to ensure single-use semantics.

        Note: This only affects auto-instrumented spans. Manual Observe spans
        are not affected.

        Args:
            input_payload: Input data to attach to the span. Will be JSON-serialized
                if not already a string.
            output_payload: Output data to attach to the span. Will be JSON-serialized
                if not already a string.
            prompt: Prompt object to extract metadata from (slug, version, provider, model).
                If provided along with variables, prompt variables will be merged with
                the explicit variables parameter.
            metadata: Custom metadata dictionary. Will be JSON-serialized and stored
                as basalt.span.metadata attribute.
            variables: Variables dictionary. Will be JSON-serialized and stored as
                basalt.span.variables attribute. If prompt is also provided, variables
                will be merged (explicit variables override prompt variables).

        Example:
            ```python
            # Inject data before an auto-instrumented LLM call
            Observe.inject_for_auto_instrumentation(
                input_payload={"query": "What is AI?"},
                prompt=my_prompt,
                metadata={"session_id": "abc123"}
            )

            # Next auto-instrumented call will have this data attached
            response = client.chat.completions.create(...)
            ```
        """
        from opentelemetry import context as otel_context
        from opentelemetry.context import attach

        from .processors import (
            PENDING_INJECT_INPUT_KEY,
            PENDING_INJECT_METADATA_KEY,
            PENDING_INJECT_OUTPUT_KEY,
            PENDING_INJECT_PROMPT_KEY,
            PENDING_INJECT_VARIABLES_KEY,
        )

        # Get current context
        ctx = otel_context.get_current()

        # Set input payload
        if input_payload is not None:
            ctx = otel_context.set_value(PENDING_INJECT_INPUT_KEY, input_payload, ctx)

        # Set output payload
        if output_payload is not None:
            ctx = otel_context.set_value(PENDING_INJECT_OUTPUT_KEY, output_payload, ctx)

        # Set variables
        if variables is not None:
            ctx = otel_context.set_value(PENDING_INJECT_VARIABLES_KEY, variables, ctx)

        # Set metadata
        if metadata is not None:
            ctx = otel_context.set_value(PENDING_INJECT_METADATA_KEY, metadata, ctx)

        # Extract and set prompt metadata
        if prompt is not None:
            prompt_data = {
                "slug": prompt.slug,
                "version": prompt.version,
                "provider": prompt.model.provider,
                "model": prompt.model.model,
            }
            # Merge prompt variables with explicit variables
            # Explicit variables take precedence over prompt variables
            merged_variables = {**(prompt.variables or {}), **(variables or {})}
            if merged_variables:
                ctx = otel_context.set_value(PENDING_INJECT_VARIABLES_KEY, merged_variables, ctx)

            ctx = otel_context.set_value(PENDING_INJECT_PROMPT_KEY, prompt_data, ctx)

        # Attach the new context
        attach(ctx)

    @staticmethod
    def _set_evaluation_config(config: EvaluationConfig | dict[str, Any]) -> None:
        """Set evaluation configuration for the root span.

        Note: This can only be called on root spans (StartSpanHandle).
        Calling from a child span will log a warning and have no effect.
        """
        handle = get_current_span_handle()
        if handle:
            if not isinstance(handle, StartSpanHandle):
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    "_set_evaluation_config() can only be called on root spans (StartSpanHandle). "
                    "This call will be ignored."
                )
                return
            handle.set_evaluation_config(config)

    @staticmethod
    def add_evaluator(slug: str) -> None:
        """Add an evaluator to the current span."""
        handle = get_current_span_handle()
        if handle:
            handle.add_evaluator(slug)

    @staticmethod
    def add_evaluators(*slugs: str) -> None:
        """Add multiple evaluators to the current span."""
        handle = get_current_span_handle()
        if handle:
            handle.add_evaluators(*slugs)

    @staticmethod
    def set_identity(identity: Identity | None = None) -> None:
        """
        Set identity on the current span.

        Args:
            identity: Identity TypedDict with optional 'user' and 'organization' keys.
                Each key should contain a dict with 'id' (required) and 'name' (optional).

        Example:
            >>> Observe.set_identity({
            ...     "user": {"id": "user-123", "name": "John Doe"},
            ...     "organization": {"id": "org-456", "name": "ACME Corp"}
            ... })
        """
        handle = get_current_span_handle()
        if handle:
            handle.set_identity(identity)


class AsyncStartObserve:
    """
    Async version of StartObserve for use with async with statements.
    Must be used as the root span of a trace in async contexts.
    """

    def __init__(
        self,
        feature_slug: str,
        name: str,
        *,
        identity: Identity | Callable[..., Identity | None] | None = None,
        evaluate_config: EvaluationConfig | None = None,
        evaluators: Sequence[Any] | None = None,
        experiment: str | Experiment | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        # Validate feature_slug is provided and non-empty
        if not feature_slug or not isinstance(feature_slug, str) or not feature_slug.strip():
            raise ValueError(
                "feature_slug is required and must be a non-empty string. "
                "Please provide a valid feature identifier (e.g., 'user_authentication', 'payment_processing')."
            )

        # Validate name is provided and non-empty
        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError(
                "name is required and must be a non-empty string. "
                "Please provide a descriptive name for this span (e.g., 'async_workflow', 'async_operation')."
            )

        self.name = name.strip()
        self.feature_slug = feature_slug.strip()
        self.identity_resolver = identity
        self.evaluate_config = evaluate_config
        self.evaluators = evaluators
        self.experiment = experiment
        self._metadata = metadata
        self._span_handle: StartSpanHandle | None = None
        self._ctx_manager = None

    async def __aenter__(self) -> StartSpanHandle:
        span_name = self.name

        # Resolve identity
        user_identity, org_identity = resolve_identity_payload(self.identity_resolver, None)

        # Initialize async context manager
        self._ctx_manager = _async_with_span_handle(
            name=span_name,
            attributes=None,
            tracer_name="basalt.observability",
            handle_cls=StartSpanHandle,
            span_type="basalt_trace",
            user=user_identity,
            organization=org_identity,
            evaluators=self.evaluators,
            feature_slug=self.feature_slug,
            metadata=self._metadata,
        )
        span = await self._ctx_manager.__aenter__()
        # Type assertion: we know this is StartSpanHandle since we passed it as handle_cls
        assert isinstance(span, StartSpanHandle)
        self._span_handle = span

        # Set evaluation config if provided
        if self.evaluate_config is not None:
            self._span_handle.set_evaluation_config(self.evaluate_config)

        # Attach experiment if provided (only on root span, which this is)
        if self.experiment:
            self._apply_experiment(self._span_handle)

        return self._span_handle

    async def __aexit__(self, exc_type, exc_value, traceback):
        if self._ctx_manager:
            return await self._ctx_manager.__aexit__(exc_type, exc_value, traceback)
        return None

    def _apply_experiment(self, span: StartSpanHandle | None) -> None:
        """Apply experiment metadata to the provided span."""
        if span is None or not self.experiment:
            return

        exp_id: str | None = None

        # Handle string ID case
        if isinstance(self.experiment, str):
            exp_id = self.experiment
        # Check if it's an Experiment dataclass by looking for the 'id' attribute
        elif hasattr(self.experiment, "id"):
            exp_id = self.experiment.id  # type: ignore[attr-defined]

        if not exp_id:
            return  # Must have an id to attach

        span.set_experiment(
            experiment=exp_id,
        )


class AsyncObserve:
    """
    Async version of Observe for use with async with statements.
    Acts as an async context manager for creating spans in async code.
    """

    def __init__(
        self,
        name: str,
        kind: ObserveKind | str = ObserveKind.SPAN,
        *,
        metadata: dict[str, Any] | None = None,
        evaluators: Sequence[Any] | None = None,
        input: Any = None,
        output: Any = None,
        variables: dict[str, Any] | None = None,
        prompt: Prompt | None = None,
    ):
        # Validate name is provided and non-empty
        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError(
                "name is required and must be a non-empty string. "
                "Please provide a descriptive name for this span (e.g., 'async_operation', 'async_fetch')."
            )

        self.name = name.strip()
        self.kind = kind
        self._metadata = metadata
        self.evaluators = evaluators
        self.input_resolver = input
        self.output_resolver = output
        self.variables_resolver = variables
        self.prompt = prompt
        self._span_handle: SpanHandle | None = None
        self._ctx_manager = None

    @staticmethod
    def _get_config_for_kind(kind_str: str):
        """Return handle class, tracer name, and default resolvers for the kind."""
        if kind_str == "generation":
            return (
                LLMSpanHandle,
                "basalt.observability.generation",
                default_generation_input,
                default_generation_variables,
            )
        elif kind_str == "retrieval":
            return (
                RetrievalSpanHandle,
                "basalt.observability.retrieval",
                default_retrieval_input,
                default_retrieval_variables,
            )
        elif kind_str == "tool":
            return (
                ToolSpanHandle,
                "basalt.observability.tool",
                None,
                None,
            )
        elif kind_str == "function":
            return (
                FunctionSpanHandle,
                "basalt.observability.function",
                None,
                None,
            )
        elif kind_str == "event":
            return (
                EventSpanHandle,
                "basalt.observability.event",
                None,
                None,
            )
        else:
            return (
                SpanHandle,
                "basalt.observability",
                None,
                None,
            )

    async def __aenter__(self) -> SpanHandle:
        span_name = self.name

        if isinstance(self.kind, ObserveKind):
            kind_str = self.kind.value
        else:
            kind_str = str(self.kind).lower()
            # Validate that the string kind is valid
            valid_kinds = {k.value for k in ObserveKind}
            if kind_str not in valid_kinds:
                raise ValueError(f"Invalid kind '{kind_str}'. Must be one of: {', '.join(sorted(valid_kinds))}")

        # Reject ROOT kind
        if kind_str == ObserveKind.ROOT.value:
            raise ValueError(
                f"Cannot use kind='{ObserveKind.ROOT.value}' with AsyncObserve. "
                f"Use AsyncStartObserve (async_start_observe) for root spans."
            )

        handle_cls, tracer_name, _, _ = self._get_config_for_kind(kind_str)

        # Process prompt parameter if provided
        if self.prompt is not None:
            # Prepare prompt metadata for span attributes
            import json

            prompt_metadata = {
                "basalt.prompt.slug": self.prompt.slug,
                "basalt.prompt.version": self.prompt.version,
                "basalt.prompt.model.provider": self.prompt.model.provider,
                "basalt.prompt.model.model": self.prompt.model.model,
            }

            # Store prompt.variables separately if available (must serialize to JSON for OpenTelemetry)
            if self.prompt.variables:
                prompt_metadata["basalt.prompt.variables"] = json.dumps(self.prompt.variables)

            # Merge with existing metadata
            if self._metadata is None:
                self._metadata = prompt_metadata
            else:
                self._metadata = {**self._metadata, **prompt_metadata}

        # Check for root span
        from opentelemetry import context as otel_context

        if not otel_context.get_value(ROOT_SPAN_CONTEXT_KEY):
            import logging

            logger = logging.getLogger(__name__)
            logger.warning("AsyncObserve used without a preceding async_start_observe. This may lead to missing trace context.")

        self._ctx_manager = _async_with_span_handle(
            name=span_name,
            attributes=None,
            tracer_name=tracer_name,
            handle_cls=handle_cls,
            span_type=kind_str,
            evaluators=self.evaluators,
            metadata=self._metadata,
        )
        self._span_handle = await self._ctx_manager.__aenter__()
        return self._span_handle

    async def __aexit__(self, exc_type, exc_value, traceback):
        if self._ctx_manager:
            return await self._ctx_manager.__aexit__(exc_type, exc_value, traceback)
        return None


# Singleton instance
observe = Observe
start_observe = StartObserve
async_observe = AsyncObserve
async_start_observe = AsyncStartObserve
