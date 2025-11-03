"""
Example: Comprehensive telemetry and observability patterns with the Basalt SDK.

This example demonstrates:
- Basic telemetry configuration with TelemetryConfig
- All specialized span types (generation, retrieval, tool, event, function)
- @evaluator decorator with sample_rate and metadata propagation
- Integration with auto-instrumented LLM libraries (OpenAI SDK)
- Experiment tracking with span-level configuration
- Custom OTLP exporter configuration

This file uses primarily mock data for easy execution without API keys.

Requirements:
- Basalt SDK installed
- (Optional) OpenAI SDK for demonstrating auto-instrumentation: `pip install openai`
- (Optional) Custom OTLP collector endpoint for testing custom exporters

Install with:
    pip install basalt-sdk
    # Optional: for auto-instrumentation demo
    pip install openai

Custom Provider Instrumentation
--------------------------------
To add instrumentation for providers not included by default, install the
appropriate OpenTelemetry instrumentation package and instrument it manually:

    from opentelemetry.instrumentation.custom_provider import CustomProviderInstrumentor

    # Initialize your Basalt client first
    basalt = Basalt(api_key="your-key", telemetry_config=...)

    # Then instrument your custom provider
    CustomProviderInstrumentor().instrument()

Supported providers include: openai, anthropic, google_genai, cohere, bedrock,
vertexai, together, replicate, langchain, llamaindex, haystack
"""

from __future__ import annotations

import logging
import os
import time

from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from basalt import Basalt, TelemetryConfig
from basalt.observability.context_managers import (
    trace_event,
    trace_function,
    trace_retrieval,
    trace_span,
    trace_tool,
)
from basalt.observability.decorators import evaluator
from basalt.observability.decorators import trace_generation as trace_generation_decorator

# --- 1. Basic Telemetry Configuration ---

def build_default_client() -> Basalt:
    """
    Build a client with default telemetry configuration.

    By default, traces are sent to Basalt's OTEL collector:
    - Production: https://otel.getbasalt.ai/v1/traces
    - Development: http://localhost:4318/v1/traces

    This configuration enables LLM provider instrumentation for all available providers.
    """
    telemetry = TelemetryConfig(
        service_name="telemetry-example",
        environment="development",
        enable_llm_instrumentation=True,
        llm_trace_content=False,  # omit prompt/completion bodies from traces
    )
    return Basalt(
        api_key=os.getenv("BASALT_API_KEY", "fake-key"),
        telemetry_config=telemetry,
        trace_user={"id": "demo-user-123"},
    )


def build_custom_exporter_client() -> Basalt:
    """
    Build a client with a custom OTLP exporter endpoint.

    Useful for testing with local OTLP collectors or custom observability backends.

    IMPORTANT: When providing a custom exporter, you must manually add authentication
    headers if your collector requires authentication. The SDK only adds headers
    automatically when building the default exporter from environment variables.
    """
    # Get API key for authentication
    api_key = os.getenv("BASALT_API_KEY", "fake-key")

    # Create a custom exporter with authentication headers
    # For local development without authentication, you can omit the headers parameter
    exporter = OTLPSpanExporter(
        endpoint="http://127.0.0.1:4317",
        headers={"authorization": f"Bearer {api_key}"},  # Add auth headers manually
        insecure=True,
        timeout=10
    )

    telemetry = TelemetryConfig(
        service_name="telemetry-example-custom",
        environment="development",
        exporter=exporter,
        enable_llm_instrumentation=True,
        llm_trace_content=True,
        llm_enabled_providers=["openai"],  # Only instrument OpenAI
    )
    return Basalt(
        api_key=api_key,
        telemetry_config=telemetry,
        trace_user={"id": "demo-user-123"},
    )


# Initialize the client
basalt_client = build_default_client()


# --- 2. Evaluator Decorator with Sample Rate and Metadata ---

@evaluator(
    slugs=["response-quality", "hallucination-check"],
    sample_rate=0.8,  # Only evaluate 80% of calls
    metadata=lambda query, results_count, **kwargs: {
        "query_length": len(query),
        "results_returned": results_count,
        "source": "vector_db",
    }
)
def search_knowledge_base(query: str, top_k: int = 5) -> list[dict]:
    """
    Mock vector database search with evaluator decorator.

    The @evaluator decorator will:
    - Attach evaluator slugs ["response-quality", "hallucination-check"] to spans
    - Set sample_rate=0.8 in the span's evaluator config
    - Resolve and attach metadata (query_length, results_returned, source) to the span

    This works automatically via the BasaltCallEvaluatorProcessor when any
    instrumented span is created within this function's context!
    """
    # Mock search results
    results = [
        {"id": f"doc-{i}", "score": 0.9 - (i * 0.1), "content": f"Result {i} for query: {query}"}
        for i in range(min(top_k, 3))
    ]
    logging.info(f"Found {len(results)} results for query: {query}")
    return results


# --- 3. All Specialized Span Types ---

@trace_generation_decorator(name="llm.mock_completion")
def mock_llm_call(prompt: str, model: str = "gpt-4") -> dict:
    """
    Mock LLM call using @trace_generation decorator.

    This decorator creates a span with type='llm' and captures LLM-specific attributes.
    """
    return {
        "choices": [{"message": {"content": f"Mock response to: {prompt[:50]}..."}}],
        "model": model,
        "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": 20},
    }


def demonstrate_retrieval_span():
    """Demonstrate trace_retrieval context manager with RetrievalSpanHandle."""
    with trace_retrieval("vector_db.search") as span:
        query = "What is observability?"
        span.set_query(query)
        span.set_top_k(10)
        span.set_input({"query": query, "filters": {"type": "documentation"}})

        # Mock retrieval operation
        time.sleep(0.05)  # Simulate DB query
        results = [{"id": i, "score": 0.9 - (i * 0.1)} for i in range(5)]

        span.set_results_count(len(results))
        span.set_output({"results": results, "total_found": len(results)})
        logging.info(f"Retrieved {len(results)} documents")


def demonstrate_tool_span():
    """Demonstrate trace_tool context manager with ToolSpanHandle."""
    with trace_tool("weather_api.get_forecast") as span:
        span.set_tool_name("get_weather_forecast")
        span.set_input({"location": "San Francisco", "days": 3})

        # Mock tool execution
        time.sleep(0.03)  # Simulate API call
        result = {
            "location": "San Francisco",
            "forecast": ["Sunny", "Cloudy", "Rainy"],
            "temperatures": [72, 68, 65],
        }

        span.set_output(result)
        logging.info(f"Tool executed: {span.attributes.get('tool.name')}")


def demonstrate_event_span():
    """Demonstrate trace_event context manager with EventSpanHandle."""
    with trace_event("user.action") as span:
        span.set_event_type("button_click")
        span.set_payload({
            "button_id": "submit_form",
            "page": "/dashboard",
            "timestamp": time.time(),
        })
        span.set_input({"action": "form_submission"})
        span.set_output({"status": "success", "form_id": "user-profile-123"})
        logging.info("User event tracked")


def demonstrate_function_span():
    """Demonstrate trace_function context manager for compute operations."""
    with trace_function("data.process_batch") as span:
        span.set_input({"batch_size": 100, "data_type": "user_events"})

        # Mock data processing
        processed_count = 0
        for i in range(100):
            # Simulate processing
            if i % 20 == 0:
                span.add_event(f"processed_{i}_items")
            processed_count += 1

        span.set_output({"processed": processed_count, "errors": 0})
        span.set_attribute("processing.duration_ms", 150)
        logging.info(f"Processed {processed_count} items")


# --- 4. Experiment Tracking ---

def demonstrate_experiment_tracking():
    """
    Demonstrate experiment tracking with span-level configuration.

    Uses span.set_experiment() and span.add_evaluator() for clear,
    localized configuration without global state mutation.
    """
    # Variant A: GPT-4o
    with trace_span("experiment.run_variant_a") as span:
        # Set experiment metadata directly on the span
        span.set_experiment("exp-456", name="Model Comparison A/B Test")
        span.add_evaluator("consistency-check")
        span.set_input({"variant": "A", "model": "gpt-4o"})

        # Simulate experiment variant A
        result_a = mock_llm_call("Explain quantum computing", model="gpt-4o")

        span.set_output({"variant": "A", "result": result_a})
        span.set_attribute("experiment.variant", "A")

    # Variant B: GPT-5-mini
    with trace_span("experiment.run_variant_b") as span:
        # Each span can have its own experiment configuration
        span.set_experiment("exp-456", name="Model Comparison A/B Test")
        span.add_evaluator("consistency-check")
        span.set_input({"variant": "B", "model": "gpt-5-mini"})

        # Simulate experiment variant B
        result_b = mock_llm_call("Explain quantum computing", model="gpt-5-mini")

        span.set_output({"variant": "B", "result": result_b})
        span.set_attribute("experiment.variant", "B")

    logging.info("Experiment tracking completed")


# --- 5. Integration with Auto-Instrumented Libraries (OpenAI SDK) ---

def demonstrate_openai_integration():
    """
    Demonstrate evaluator integration with auto-instrumented OpenAI SDK.

    The @evaluator decorator will automatically attach evaluators to the
    OpenAI SDK's auto-instrumented spans via the BasaltCallEvaluatorProcessor.
    """
    try:
        from openai import OpenAI
    except ImportError:
        logging.warning("OpenAI SDK not installed, skipping auto-instrumentation demo")
        return

    @evaluator(
        slugs=["openai-quality", "toxicity"],
        sample_rate=1.0,
        metadata=lambda prompt, **kwargs: {
            "prompt_length": len(prompt),
            "context": "customer_support",
        }
    )
    def call_openai_with_evaluators(prompt: str) -> str:
        """
        Call OpenAI API with automatic evaluator attachment.

        The evaluators and metadata will be automatically attached to the
        span created by the OpenAI instrumentation!
        """
        # Mock OpenAI client (replace with real API key for actual calls)
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "fake-key"))

        # This call is auto-instrumented by opentelemetry-instrumentation-openai
        # The @evaluator decorator ensures evaluators are attached to this span!
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
            )
            return response.choices[0].message.content
        except Exception as exc:
            logging.warning(f"OpenAI API call failed (expected with fake key): {exc}")
            return "Mock response (API call failed)"

    # Use the search function which has evaluators attached
    with trace_span("openai.integration_demo") as span:
        # First, do a knowledge base search (with evaluators)
        search_results = search_knowledge_base("How do I configure telemetry?", top_k=3)
        span.add_event("knowledge_base_searched")

        # Then call OpenAI with context from search (with evaluators)
        context = " ".join([r["content"] for r in search_results])
        prompt = f"Based on this context: {context}, answer: How do I configure telemetry?"

        response = call_openai_with_evaluators(prompt)
        span.set_output({"response": response, "source": "openai+kb"})
        logging.info("OpenAI integration demo completed")


# --- 6. Main Workflow ---

def main():
    """
    Demonstrate comprehensive telemetry patterns.

    This example showcases:
    1. Basic telemetry setup
    2. Evaluator decorator with sample_rate and metadata
    3. All specialized span types
    4. Experiment tracking
    5. Integration with auto-instrumented libraries
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Wrap the entire workflow in a single root span
    with trace_span(
        "workflow.telemetry_demo",
        attributes={
            "workflow.type": "comprehensive_demo",
            "service": "telemetry-example",
        }
    ) as root_span:

        root_span.set_input({"demo_type": "full_telemetry_showcase"})

        # Section 1: Evaluator decorator demo
        logging.info("\n--- Section 1: Evaluator Decorator with Metadata ---")
        search_knowledge_base("What is OpenTelemetry?", top_k=5)
        root_span.add_event("knowledge_base_search_completed")

        # Section 2: All specialized span types
        logging.info("\n--- Section 2: Specialized Span Types ---")
        demonstrate_retrieval_span()
        demonstrate_tool_span()
        demonstrate_event_span()
        demonstrate_function_span()
        root_span.add_event("specialized_spans_completed")

        # Section 3: Experiment tracking
        logging.info("\n--- Section 3: Experiment Tracking ---")
        demonstrate_experiment_tracking()
        root_span.add_event("experiment_tracking_completed")

        # Section 4: OpenAI integration (optional)
        logging.info("\n--- Section 4: Auto-Instrumented Library Integration ---")
        demonstrate_openai_integration()
        root_span.add_event("openai_integration_completed")

        # Section 5: Mock LLM call with decorator
        logging.info("\n--- Section 5: Mock LLM Call ---")
        mock_llm_call("Explain the importance of observability")
        root_span.add_event("mock_llm_call_completed")

        root_span.set_output({
            "status": "success",
            "sections_completed": 5,
            "total_operations": 10,
        })

    # Clean up - shutdown client to flush all traces
    logging.info("\nShutting down and flushing telemetry...")
    basalt_client.shutdown()
    logging.info("Demo completed successfully!")


if __name__ == "__main__":
    main()
