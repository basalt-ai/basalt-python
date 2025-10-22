"""
OpenAI instrumentation for the Basalt SDK.

This module provides automatic instrumentation for the OpenAI Python SDK,
capturing traces and telemetry data for chat completions and other operations.
"""
from __future__ import annotations

import logging
from typing import Any

import wrapt

from ..tracing.provider import get_tracer
from .semconv import GenAIAttributes

logger = logging.getLogger(__name__)


class OpenAIInstrumentor:
    """
    Instrumentor for the OpenAI Python SDK.

    This class uses wrapt to patch OpenAI methods and automatically create
    spans with GenAI semantic conventions.

    Example:
        ```python
        from basalt.instrumentation import OpenAIInstrumentor

        # Instrument OpenAI
        instrumentor = OpenAIInstrumentor()
        instrumentor.instrument()

        # Now all OpenAI calls will be traced
        import openai
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        ```
    """

    def __init__(self, tracer_provider: Any = None):
        """
        Initialize the OpenAI instrumentor.

        Args:
            tracer_provider: Optional tracer provider. If not provided, uses the global provider.
        """
        self._tracer_provider = tracer_provider
        self._is_instrumented = False
        self._original_methods: dict[str, Any] = {}

    def instrument(self) -> None:
        """
        Instrument the OpenAI SDK.

        This patches the OpenAI methods to add tracing capabilities.
        """
        if self._is_instrumented:
            logger.warning("OpenAI is already instrumented")
            return

        try:
            import openai
        except ImportError:
            logger.warning("OpenAI SDK not installed, skipping instrumentation")
            return

        try:
            # Patch chat completions (sync)
            if hasattr(openai, "OpenAI"):
                wrapt.wrap_function_wrapper(
                    "openai.resources.chat.completions",
                    "Completions.create",
                    self._wrap_chat_completion,
                )

            # Patch chat completions (async)
            if hasattr(openai, "AsyncOpenAI"):
                wrapt.wrap_function_wrapper(
                    "openai.resources.chat.completions",
                    "AsyncCompletions.create",
                    self._wrap_chat_completion_async,
                )

            self._is_instrumented = True
            logger.info("OpenAI SDK instrumented successfully")

        except Exception as e:
            logger.error(f"Failed to instrument OpenAI SDK: {e}")

    def uninstrument(self) -> None:
        """
        Remove instrumentation from the OpenAI SDK.

        Note: wrapt doesn't provide a built-in way to unwrap, so this is a no-op.
        Restart the application to remove instrumentation.
        """
        if not self._is_instrumented:
            return

        logger.warning(
            "OpenAI uninstrumentation is not fully supported. "
            "Restart the application to remove instrumentation."
        )
        self._is_instrumented = False

    def _wrap_chat_completion(
        self,
        wrapped: Any,
        instance: Any,
        args: tuple,
        kwargs: dict,
    ) -> Any:
        """
        Wrapper for synchronous chat.completions.create.

        Args:
            wrapped: The original function
            instance: The instance (self) of the class
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            The result of the wrapped function
        """
        tracer = get_tracer(__name__)

        # Extract parameters
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens")
        top_p = kwargs.get("top_p")
        frequency_penalty = kwargs.get("frequency_penalty")
        presence_penalty = kwargs.get("presence_penalty")
        stop = kwargs.get("stop")

        # Create request attributes
        request_attrs = GenAIAttributes.create_request_attributes(
            system="openai",
            model=model,
            operation="chat.completions",
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop_sequences=stop if isinstance(stop, list) else ([stop] if stop else None),
        )

        with tracer.start_as_current_span(
            f"openai.chat.completions.create",
            attributes=request_attrs,
        ) as span:
            try:
                # Call the original function
                result = wrapped(*args, **kwargs)

                # Extract response attributes
                response_attrs = self._extract_response_attributes(result)
                for key, value in response_attrs.items():
                    span.set_attribute(key, value)

                return result

            except Exception as e:
                # The span will automatically record the exception
                raise

    def _wrap_chat_completion_async(
        self,
        wrapped: Any,
        instance: Any,
        args: tuple,
        kwargs: dict,
    ) -> Any:
        """
        Wrapper for asynchronous chat.completions.create.

        Args:
            wrapped: The original function
            instance: The instance (self) of the class
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            The result of the wrapped function (coroutine)
        """
        tracer = get_tracer(__name__)

        # Extract parameters
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens")
        top_p = kwargs.get("top_p")
        frequency_penalty = kwargs.get("frequency_penalty")
        presence_penalty = kwargs.get("presence_penalty")
        stop = kwargs.get("stop")

        # Create request attributes
        request_attrs = GenAIAttributes.create_request_attributes(
            system="openai",
            model=model,
            operation="chat.completions",
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop_sequences=stop if isinstance(stop, list) else ([stop] if stop else None),
        )

        async def async_wrapper():
            with tracer.start_as_current_span(
                f"openai.chat.completions.create",
                attributes=request_attrs,
            ) as span:
                try:
                    # Call the original async function
                    result = await wrapped(*args, **kwargs)

                    # Extract response attributes
                    response_attrs = self._extract_response_attributes(result)
                    for key, value in response_attrs.items():
                        span.set_attribute(key, value)

                    return result

                except Exception as e:
                    # The span will automatically record the exception
                    raise

        return async_wrapper()

    def _extract_response_attributes(self, response: Any) -> dict[str, Any]:
        """
        Extract response attributes from an OpenAI response.

        Args:
            response: The OpenAI API response

        Returns:
            Dictionary of response attributes
        """
        attrs: dict[str, Any] = {}

        try:
            # Extract response ID
            if hasattr(response, "id"):
                attrs["gen_ai.response.id"] = response.id

            # Extract model
            if hasattr(response, "model"):
                attrs["gen_ai.response.model"] = response.model

            # Extract usage information
            if hasattr(response, "usage"):
                usage = response.usage
                if hasattr(usage, "prompt_tokens"):
                    attrs["gen_ai.usage.input_tokens"] = usage.prompt_tokens
                if hasattr(usage, "completion_tokens"):
                    attrs["gen_ai.usage.output_tokens"] = usage.completion_tokens

            # Extract finish reasons
            if hasattr(response, "choices") and response.choices:
                finish_reasons = []
                for choice in response.choices:
                    if hasattr(choice, "finish_reason") and choice.finish_reason:
                        finish_reasons.append(choice.finish_reason)
                if finish_reasons:
                    attrs["gen_ai.response.finish_reasons"] = finish_reasons

        except Exception as e:
            logger.warning(f"Failed to extract response attributes: {e}")

        return attrs

    @property
    def is_instrumented(self) -> bool:
        """Check if OpenAI is currently instrumented."""
        return self._is_instrumented
