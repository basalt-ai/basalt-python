"""Tests for experiment propagation across Basalt and auto-instrumented spans."""

from __future__ import annotations

import asyncio

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from basalt.observability import observe, start_observe
from basalt.observability.decorators import ObserveKind
from basalt.observability.processors import (
    BasaltAutoInstrumentationProcessor,
    BasaltCallEvaluatorProcessor,
    BasaltContextProcessor,
    BasaltShouldEvaluateProcessor,
)
from basalt.observability.semconv import BasaltExperiment, BasaltSpan


@pytest.fixture(scope="function")
def setup_experiment_tracer():
    """Install Basalt processors + in-memory exporter for experiment propagation tests."""
    import opentelemetry.trace as trace_module

    original_provider = getattr(trace_module, "_TRACER_PROVIDER", None)
    original_set_once = getattr(trace_module, "_TRACER_PROVIDER_SET_ONCE", None)

    from opentelemetry.util._once import Once

    trace_module._TRACER_PROVIDER_SET_ONCE = Once()
    trace_module._TRACER_PROVIDER = None

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    provider.add_span_processor(BasaltContextProcessor())
    provider.add_span_processor(BasaltCallEvaluatorProcessor())
    provider.add_span_processor(BasaltShouldEvaluateProcessor())
    provider.add_span_processor(BasaltAutoInstrumentationProcessor())

    trace.set_tracer_provider(provider)

    yield exporter

    exporter.clear()
    provider.shutdown()

    trace_module._TRACER_PROVIDER_SET_ONCE = Once()
    trace_module._TRACER_PROVIDER = None

    if original_provider and not isinstance(original_provider, trace.ProxyTracerProvider):
        trace.set_tracer_provider(original_provider)

    if original_set_once is not None:
        trace_module._TRACER_PROVIDER_SET_ONCE = original_set_once


class TestExperimentPropagation:
    """Test suite for experiment attribute propagation to all child spans."""

    def test_experiment_id_propagates_to_child_spans(self, setup_experiment_tracer):
        """Test that experiment ID propagates to all nested basalt spans."""
        exporter = setup_experiment_tracer

        with start_observe(
            name="root",
            feature_slug="test-feature",
            experiment="exp_123",
        ):
            with observe(name="child1", kind=ObserveKind.FUNCTION):
                with observe(name="grandchild", kind=ObserveKind.FUNCTION):
                    pass

            with observe(name="child2", kind=ObserveKind.GENERATION):
                pass

        spans = exporter.get_finished_spans()
        span_by_name = {span.name: span for span in spans}

        expected_names = {"root", "child1", "grandchild", "child2"}
        assert expected_names.issubset(span_by_name.keys()), f"Missing spans: {expected_names - set(span_by_name.keys())}"

        for name in expected_names:
            span = span_by_name[name]
            assert span.attributes.get(BasaltExperiment.ID) == "exp_123", (
                f"Span '{name}' missing experiment ID propagation. Expected 'exp_123', got {span.attributes.get(BasaltExperiment.ID)}"
            )

    def test_experiment_dict_propagates_all_fields(self, setup_experiment_tracer):
        """Test that a rich experiment (with name and feature_slug) propagates all fields."""
        exporter = setup_experiment_tracer

        # Simulate passing an Experiment dataclass by using the dict payload format
        # (this is what _build_experiment_payload produces from an Experiment object)
        from basalt.experiments.models import Experiment

        experiment = Experiment(
            id="exp_456",
            name="My Test Experiment",
            feature_slug="support-ticket",
            created_at="2025-01-01T00:00:00Z",
        )

        with start_observe(
            name="root",
            feature_slug="support-ticket",
            experiment=experiment,
        ):
            with observe(name="child", kind=ObserveKind.FUNCTION):
                pass

        spans = exporter.get_finished_spans()
        span_by_name = {span.name: span for span in spans}

        for name in ("root", "child"):
            span = span_by_name[name]
            assert span.attributes.get(BasaltExperiment.ID) == "exp_456", f"Span '{name}' missing experiment.id"
            assert span.attributes.get(BasaltExperiment.NAME) == "My Test Experiment", f"Span '{name}' missing experiment.name"
            assert span.attributes.get(BasaltExperiment.FEATURE_SLUG) == "support-ticket", f"Span '{name}' missing experiment.feature_slug"

    def test_experiment_propagates_to_auto_instrumented_spans(self, setup_experiment_tracer):
        """Test that experiment propagates to auto-instrumented (e.g. OpenAI) spans."""
        exporter = setup_experiment_tracer

        tracer_openai = trace.get_tracer("opentelemetry.instrumentation.openai.v1")
        tracer_httpx = trace.get_tracer("opentelemetry.instrumentation.httpx")

        with start_observe(
            name="root",
            feature_slug="test-feature",
            experiment="exp_auto",
        ):
            with tracer_openai.start_as_current_span("openai.call"):
                pass

            with tracer_httpx.start_as_current_span("httpx.request"):
                pass

        spans = exporter.get_finished_spans()
        span_by_name = {span.name: span for span in spans}

        expected_with_experiment = {"root", "openai.call", "httpx.request"}
        assert expected_with_experiment.issubset(span_by_name.keys()), f"Missing spans: {expected_with_experiment - set(span_by_name.keys())}"

        for name in expected_with_experiment:
            span = span_by_name[name]
            assert span.attributes.get(BasaltExperiment.ID) == "exp_auto", (
                f"Span '{name}' missing experiment ID propagation. Got {span.attributes.get(BasaltExperiment.ID)}"
            )

    def test_experiment_propagates_to_fastapi_parent_span(self, setup_experiment_tracer):
        """Test that experiment is backfilled onto a non-Basalt parent span (e.g. FastAPI)."""
        exporter = setup_experiment_tracer

        tracer_fastapi = trace.get_tracer("opentelemetry.instrumentation.fastapi")

        with tracer_fastapi.start_as_current_span("fastapi.request"):
            with start_observe(
                name="handler",
                feature_slug="support-ticket",
                experiment="exp_backfill",
            ):
                with observe(name="child", kind=ObserveKind.FUNCTION):
                    pass

        spans = exporter.get_finished_spans()
        span_by_name = {span.name: span for span in spans}

        # All spans including the fastapi parent should have the experiment
        for name in ("fastapi.request", "handler", "child"):
            span = span_by_name[name]
            assert span.attributes.get(BasaltExperiment.ID) == "exp_backfill", (
                f"Span '{name}' missing experiment ID. Got {span.attributes.get(BasaltExperiment.ID)}"
            )

    def test_experiment_propagates_to_decorated_functions(self, setup_experiment_tracer):
        """Test that experiment propagates to @observe decorated functions."""
        exporter = setup_experiment_tracer

        @observe(name="decorated_fn", kind=ObserveKind.FUNCTION)
        def my_function():
            return "result"

        with start_observe(
            name="root",
            feature_slug="test-feature",
            experiment="exp_decorated",
        ):
            my_function()

        spans = exporter.get_finished_spans()
        span_by_name = {span.name: span for span in spans}

        assert "decorated_fn" in span_by_name, "decorated_fn span not found"
        fn_span = span_by_name["decorated_fn"]
        assert fn_span.attributes.get(BasaltExperiment.ID) == "exp_decorated", (
            f"Decorated function span missing experiment ID. Got {fn_span.attributes.get(BasaltExperiment.ID)}"
        )

    def test_experiment_propagates_in_async_context(self, setup_experiment_tracer):
        """Test that experiment propagates in async start_observe + async observe."""
        exporter = setup_experiment_tracer

        @observe(name="async_fn", kind=ObserveKind.FUNCTION)
        async def my_async_function():
            return "async_result"

        async def run():
            with start_observe(
                name="async_root",
                feature_slug="async-test",
                experiment="exp_async",
            ):
                await my_async_function()

        asyncio.run(run())

        spans = exporter.get_finished_spans()
        span_by_name = {span.name: span for span in spans}

        for name in ("async_root", "async_fn"):
            span = span_by_name[name]
            assert span.attributes.get(BasaltExperiment.ID) == "exp_async", (
                f"Span '{name}' missing experiment ID in async context. Got {span.attributes.get(BasaltExperiment.ID)}"
            )

    def test_no_experiment_when_not_set(self, setup_experiment_tracer):
        """Test that spans have no experiment attributes when none is set."""
        exporter = setup_experiment_tracer

        with start_observe(name="root", feature_slug="no-exp"):
            with observe(name="child", kind=ObserveKind.FUNCTION):
                pass

        spans = exporter.get_finished_spans()
        for span in spans:
            assert span.attributes.get(BasaltExperiment.ID) is None, f"Span '{span.name}' should not have experiment ID"
