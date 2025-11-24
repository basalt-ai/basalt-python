from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Sequence
from contextlib import ContextDecorator
from typing import Any, NotRequired, TypedDict, TypeVar

from opentelemetry.trace import StatusCode

from basalt.experiments.models import Experiment  # Allow passing Experiment dataclass instances

from .context_managers import (
    ROOT_SPAN_CONTEXT_KEY,
    EvaluatorConfig,
    EventSpanHandle,
    FunctionSpanHandle,
    LLMSpanHandle,
    RetrievalSpanHandle,
    SpanHandle,
    ToolSpanHandle,
    _with_span_handle,
    get_current_span,
    get_current_span_handle,
    get_root_span_handle,
    set_trace_organization,
    set_trace_user,
)
from .decorators import ObserveKind
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


class _IdentityEntity(TypedDict):
    """Identity entity with required id and optional name."""
    id: str
    name: NotRequired[str]


class IdentityDict(TypedDict, total=False):
    """
    Identity structure for user and organization tracking.

    Example:
        {
            "organization": {"id": "123", "name": "ACME"},
            "user": {"id": "456", "name": "John Doe"}
        }
    """
    organization: NotRequired[_IdentityEntity]
    user: NotRequired[_IdentityEntity]


class ExperimentDict(TypedDict):
    """Experiment metadata with required id and optional name/featureSlug."""
    id: str
    name: NotRequired[str]
    featureSlug: NotRequired[str]


class StartObserve(ContextDecorator):
    """
    Entry point for Basalt observability.
    Must be used as the root span of a trace.
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        identity: IdentityDict | Callable[..., IdentityDict | None] | None = None,
        evaluate_config: EvaluatorConfig | None = None,
        experiment: ExperimentDict | Experiment | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.name = name
        self.identity_resolver = identity
        self.evaluate_config = evaluate_config
        self.experiment = experiment
        self._metadata = metadata
        self._span_handle: SpanHandle | None = None
        self._ctx_manager = None

    def __enter__(self) -> SpanHandle:
        span_name = self.name or "basalt_trace"

        # Resolve identity
        user_identity, org_identity = resolve_identity_payload(self.identity_resolver, None)

        # Initialize context manager
        self._ctx_manager = _with_span_handle(
            name=span_name,
            attributes=self._metadata,
            tracer_name="basalt.observability",
            handle_cls=SpanHandle,
            span_type="basalt_trace",
            user=user_identity,
            organization=org_identity,
            evaluator_config=self.evaluate_config,
        )
        self._span_handle = self._ctx_manager.__enter__()

        # Attach experiment if provided (only on root span, which this is)
        if self.experiment:
            self._apply_experiment(self._span_handle)

        return self._span_handle

    def __exit__(self, exc_type, exc_value, traceback):
        if self._ctx_manager:
            return self._ctx_manager.__exit__(exc_type, exc_value, traceback)
        return None

    def __call__(self, func: F) -> F:
        if self.name is None:
            self.name = f"{func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Resolve identity from args/kwargs if needed
            bound = resolve_bound_arguments(func, args, kwargs)
            user_identity, org_identity = resolve_identity_payload(self.identity_resolver, bound)

            span_name = self.name or "basalt_trace"
            with _with_span_handle(
                name=span_name,
                attributes=self._metadata,
                tracer_name="basalt.observability",
                handle_cls=SpanHandle,
                span_type="basalt_trace",
                user=user_identity,
                organization=org_identity,
                evaluator_config=self.evaluate_config,
            ) as span:
                # Attach experiment if provided
                if self.experiment:
                    self._apply_experiment(span)

                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(StatusCode.ERROR, str(exc))
                    raise

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                bound = resolve_bound_arguments(func, args, kwargs)
                user_identity, org_identity = resolve_identity_payload(self.identity_resolver, bound)

                span_name = self.name or "basalt_trace"
                with _with_span_handle(
                    name=span_name,
                    attributes=self._metadata,
                    tracer_name="basalt.observability",
                    handle_cls=SpanHandle,
                    span_type="basalt_trace",
                    user=user_identity,
                    organization=org_identity,
                    evaluator_config=self.evaluate_config,
                ) as span:
                    if self.experiment:
                        self._apply_experiment(span)
                    try:
                        return await func(*args, **kwargs)
                    except Exception as exc:
                        span.record_exception(exc)
                        span.set_status(StatusCode.ERROR, str(exc))
                        raise

            return async_wrapper  # type: ignore

        return wrapper  # type: ignore

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _apply_experiment(self, span: SpanHandle | None) -> None:
        """Apply experiment metadata to the provided span.

        Supports either a raw experiment dict (ExperimentDict) or an
        `Experiment` dataclass instance from `basalt.experiments.models`.
        """
        if span is None or not self.experiment:
            return

        exp_id: str | None = None
        exp_name: str | None = None
        exp_feature_slug: str | None = None

        if isinstance(self.experiment, Experiment):
            exp_id = self.experiment.id
            exp_name = self.experiment.name or None
            # Dataclass uses `feature_slug`
            exp_feature_slug = self.experiment.feature_slug or None
        elif isinstance(self.experiment, dict):  # ExperimentDict
            exp_id = self.experiment.get("id")
            exp_name = self.experiment.get("name")
            # Accept multiple key styles: variant, feature_slug, featureSlug
            exp_feature_slug = (
                self.experiment.get("variant")
                or self.experiment.get("feature_slug")
                or self.experiment.get("featureSlug")
            )

        if not exp_id:
            return  # Must have an id to attach

        span.set_experiment(
            experiment_id=exp_id,
            name=exp_name,
            feature_slug=exp_feature_slug,
        )


class Observe(ContextDecorator):
    """
    Unified observability interface for Basalt.
    Acts as both a decorator and a context manager.
    """

    def __init__(
        self,
        name: str | None = None,
        kind: ObserveKind | str = ObserveKind.SPAN,
        *,
        metadata: dict[str, Any] | None = None,
        evaluators: Sequence[Any] | None = None,
        input: Any = None,
        output: Any = None,
        variables: Any = None,
    ):
        self.name = name
        self.kind = kind
        self._metadata = metadata
        self.evaluators = evaluators
        self.input_resolver = input
        self.output_resolver = output
        self.variables_resolver = variables
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
        span_name = self.name or "unknown_span"

        if isinstance(self.kind, ObserveKind):
            kind_str = self.kind.value
        else:
            kind_str = str(self.kind).lower()
            # Validate that the string kind is valid
            valid_kinds = {k.value for k in ObserveKind}
            if kind_str not in valid_kinds:
                raise ValueError(f"Invalid kind '{kind_str}'. Must be one of: {', '.join(sorted(valid_kinds))}")

        handle_cls, tracer_name, _, _ = self._get_config_for_kind(kind_str)

        # Check for root span
        from opentelemetry import context as otel_context

        if not otel_context.get_value(ROOT_SPAN_CONTEXT_KEY):
            import logging

            logger = logging.getLogger(__name__)
            logger.warning("Observe used without a preceding start_observe. This may lead to missing trace context.")

        self._ctx_manager = _with_span_handle(
            name=span_name,
            attributes=self._metadata,
            tracer_name=tracer_name,
            handle_cls=handle_cls,
            span_type=kind_str,
            evaluators=self.evaluators,
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
        if self.name is None:
            self.name = f"{func.__module__}.{func.__qualname__}"

        if isinstance(self.kind, ObserveKind):
            kind_str = self.kind.value
        else:
            kind_str = str(self.kind).lower()

        handle_cls, tracer_name, default_input, default_vars = self._get_config_for_kind(kind_str)

        # Use defaults if not provided
        input_resolver = self.input_resolver if self.input_resolver is not None else default_input
        variables_resolver = self.variables_resolver if self.variables_resolver is not None else default_vars

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            computed_attrs = resolve_attributes(self._metadata, args, kwargs)
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
                name=self.name or "unknown_span",
                attributes=computed_attrs,
                tracer_name=tracer_name,
                handle_cls=handle_cls,
                span_type=kind_str,
                input_payload=input_payload,
                variables=variables_payload,
                evaluators=pre_evaluators,
            ) as span:
                if apply_pre:
                    apply_pre(span, bound)

                try:
                    result = func(*args, **kwargs)

                    transformed = self.output_resolver(result) if self.output_resolver else result
                    span.set_output(transformed)

                    if apply_post:
                        apply_post(span, result)

                    span.set_status(StatusCode.OK)
                    return result
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_output({"error": str(exc)})
                    span.set_status(StatusCode.ERROR, str(exc))
                    raise

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                computed_attrs = resolve_attributes(self._metadata, args, kwargs)
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
                    name=self.name or "unknown_span",
                    attributes=computed_attrs,
                    tracer_name=tracer_name,
                    handle_cls=handle_cls,
                    span_type=kind_str,
                    input_payload=input_payload,
                    variables=variables_payload,
                    evaluators=pre_evaluators,
                ) as span:
                    if apply_pre:
                        apply_pre(span, bound)

                    try:
                        result = await func(*args, **kwargs)

                        transformed = self.output_resolver(result) if self.output_resolver else result
                        span.set_output(transformed)

                        if apply_post:
                            apply_post(span, result)

                        span.set_status(StatusCode.OK)
                        return result
                    except Exception as exc:
                        span.record_exception(exc)
                        span.set_output({"error": str(exc)})
                        span.set_status(StatusCode.ERROR, str(exc))
                        raise

            return async_wrapper  # type: ignore

        return wrapper  # type: ignore

    # Static Domain Methods

    @staticmethod
    def identify(user: str | dict[str, Any] | None = None, organization: str | dict[str, Any] | None = None) -> None:
        """Set the user and/or organization identity for the current context."""
        if user:
            if isinstance(user, str):
                set_trace_user(user_id=user)
            elif isinstance(user, dict):
                set_trace_user(user_id=user.get("id", "unknown"), name=user.get("name"))

        if organization:
            if isinstance(organization, str):
                set_trace_organization(organization_id=organization)
            elif isinstance(organization, dict):
                set_trace_organization(organization_id=organization.get("id", "unknown"), name=organization.get("name"))

    @staticmethod
    def metadata(data: dict[str, Any] | None = None, **kwargs) -> None:
        """Add metadata to the current span."""
        handle = get_current_span_handle()
        if not handle:
            return

        payload = {}
        if data:
            payload.update(data)
        payload.update(kwargs)

        for k, v in payload.items():
            handle.set_attribute(k, v)

    @staticmethod
    def update_metadata(data: dict[str, Any] | None = None, **kwargs) -> None:
        """Merge metadata into the current span, updating existing keys."""
        handle = get_current_span_handle()
        if not handle:
            return

        payload = {}
        if data:
            payload.update(data)
        payload.update(kwargs)

        for k, v in payload.items():
            handle.set_attribute(k, v)

    @staticmethod
    def root_span() -> SpanHandle | None:
        """Get the root span handle of the current trace.

        Returns the handle for the root span, enabling late-binding of
        identify() or metadata from deeply nested contexts.
        """
        return get_root_span_handle()

    @staticmethod
    def input(data: Any) -> None:
        """Set input data for the current span."""
        handle = get_current_span_handle()
        if handle:
            handle.set_input(data)

    @staticmethod
    def output(data: Any) -> None:
        """Set output data for the current span."""
        handle = get_current_span_handle()
        if handle:
            handle.set_output(data)

    @staticmethod
    def evaluate(evaluator: Any) -> None:
        """Attach an evaluator to the current span."""
        handle = get_current_span_handle()
        if handle:
            handle.add_evaluators(evaluator)

    @staticmethod
    def status(status: StatusCode | str, message: str | None = None) -> None:
        """Set the status of the current span."""
        span = get_current_span()
        if not span:
            return

        if isinstance(status, str):
            status_map = {"ok": StatusCode.OK, "error": StatusCode.ERROR, "unset": StatusCode.UNSET}
            code = status_map.get(status.lower(), StatusCode.UNSET)
        else:
            code = status

        span.set_status(code, description=message)

    @staticmethod
    def fail(exception: BaseException | str) -> None:
        """Record an error/exception and set status to ERROR."""
        span = get_current_span()
        if not span:
            return

        if isinstance(exception, BaseException):
            span.record_exception(exception)
            span.set_status(StatusCode.ERROR, str(exception))
        else:
            span.set_status(StatusCode.ERROR, str(exception))

    @staticmethod
    def experiment(id: str, variant: str | None = None) -> None:
        """Attach experiment context."""
        handle = get_current_span_handle()
        if handle:
            handle.set_experiment(experiment_id=id, feature_slug=variant)


# Singleton instance
observe = Observe
start_observe = StartObserve
