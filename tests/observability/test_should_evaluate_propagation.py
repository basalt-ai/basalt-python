"""Tests for should_evaluate propagation across all spans in a trace."""

from collections.abc import Sequence

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from basalt.observability import observe, start_observe
from basalt.observability.context_managers import EvaluationConfig
from basalt.observability.decorators import ObserveKind
from basalt.observability.processors import BasaltShouldEvaluateProcessor
from basalt.observability.semconv import BasaltSpan


class InMemorySpanExporter(SpanExporter):
    """Simple in-memory span exporter for testing."""

    def __init__(self) -> None:
        self.spans = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def get_finished_spans(self):
        return self.spans

    def clear(self):
        self.spans = []


@pytest.fixture(scope="function")
def setup_tracer():
    """Setup tracer with in-memory exporter for testing."""
    exporter = InMemorySpanExporter()
    provider = trace.get_tracer_provider()

    # If provider is a ProxyTracerProvider, create a real one
    if type(provider).__name__ == "ProxyTracerProvider":
        provider = TracerProvider()
        provider.add_span_processor(BasaltShouldEvaluateProcessor())
        trace.set_tracer_provider(provider)

    # Ensure BasaltShouldEvaluateProcessor is installed
    if not hasattr(provider, "_basalt_should_evaluate_installed"):
        processor = BasaltShouldEvaluateProcessor()
        provider.add_span_processor(processor)
        provider._basalt_should_evaluate_installed = True  # type: ignore[attr-defined]

    # Add the exporter processor
    span_processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(span_processor)

    yield exporter

    exporter.clear()


class TestShouldEvaluatePropagation:
    """Test suite for should_evaluate propagation."""

    def test_sample_rate_1_propagates_to_children(self, setup_tracer):
        """Test that should_evaluate=True propagates to all child spans."""
        exporter = setup_tracer

        with start_observe(
            name="parent", feature_slug="test", evaluate_config=EvaluationConfig(sample_rate=1.0)
        ):
            with observe(name="child1", kind=ObserveKind.FUNCTION):
                with observe(name="grandchild", kind=ObserveKind.FUNCTION):
                    pass

            with observe(name="child2", kind=ObserveKind.GENERATION):
                pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 4, f"Expected 4 spans, got {len(spans)}"

        # All spans should have should_evaluate=True
        for span in spans:
            assert BasaltSpan.SHOULD_EVALUATE in span.attributes, (
                f"Span {span.name} missing should_evaluate"
            )
            assert span.attributes[BasaltSpan.SHOULD_EVALUATE] is True, (
                f"Span {span.name} has should_evaluate={span.attributes[BasaltSpan.SHOULD_EVALUATE]}, expected True"
            )

    def test_sample_rate_0_propagates_to_children(self, setup_tracer):
        """Test that should_evaluate=False propagates to all child spans."""
        exporter = setup_tracer

        with start_observe(
            name="parent", feature_slug="test", evaluate_config=EvaluationConfig(sample_rate=0.0)
        ):
            with observe(name="child1", kind=ObserveKind.FUNCTION):
                with observe(name="grandchild", kind=ObserveKind.FUNCTION):
                    pass

            with observe(name="child2", kind=ObserveKind.GENERATION):
                pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 4, f"Expected 4 spans, got {len(spans)}"

        # All spans should have should_evaluate=False
        for span in spans:
            assert BasaltSpan.SHOULD_EVALUATE in span.attributes, (
                f"Span {span.name} missing should_evaluate"
            )
            assert span.attributes[BasaltSpan.SHOULD_EVALUATE] is False, (
                f"Span {span.name} has should_evaluate={span.attributes[BasaltSpan.SHOULD_EVALUATE]}, expected False"
            )

    def test_experiment_forces_true_for_all_spans(self, setup_tracer):
        """Test that experiment forces should_evaluate=True for all spans."""
        exporter = setup_tracer

        with start_observe(
            name="parent",
            feature_slug="test",
            experiment="exp_123",
            evaluate_config=EvaluationConfig(sample_rate=0.0),  # Would normally be False
        ):
            with observe(name="child1", kind=ObserveKind.FUNCTION):
                with observe(name="grandchild", kind=ObserveKind.FUNCTION):
                    pass

            with observe(name="child2", kind=ObserveKind.GENERATION):
                pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 4, f"Expected 4 spans, got {len(spans)}"

        # All spans should have should_evaluate=True due to experiment
        for span in spans:
            assert BasaltSpan.SHOULD_EVALUATE in span.attributes, (
                f"Span {span.name} missing should_evaluate"
            )
            assert span.attributes[BasaltSpan.SHOULD_EVALUATE] is True, (
                f"Span {span.name} has should_evaluate={span.attributes[BasaltSpan.SHOULD_EVALUATE]}, expected True due to experiment"
            )

    def test_experiment_overrides_sample_rate_for_all_spans(self, setup_tracer):
        """Test that experiment overrides sample_rate=0.0 for entire trace."""
        exporter = setup_tracer

        with start_observe(
            name="experiment_trace",
            feature_slug="test",
            experiment="exp_456",
            # No evaluate_config, global default is 0.0
        ):
            with observe(name="processing", kind=ObserveKind.FUNCTION):
                pass

            with observe(name="llm_call", kind=ObserveKind.GENERATION):
                with observe(name="retrieval", kind=ObserveKind.RETRIEVAL):
                    pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 4, f"Expected 4 spans, got {len(spans)}"

        # Verify all have should_evaluate=True
        for span in spans:
            assert span.attributes.get(BasaltSpan.SHOULD_EVALUATE) is True, (
                f"Span {span.name} should have should_evaluate=True with experiment"
            )

    def test_deeply_nested_spans_propagate(self, setup_tracer):
        """Test propagation through deeply nested span hierarchy."""
        exporter = setup_tracer

        with start_observe(
            name="root", feature_slug="test", evaluate_config=EvaluationConfig(sample_rate=1.0)
        ):
            with observe(name="level1", kind=ObserveKind.FUNCTION):
                with observe(name="level2", kind=ObserveKind.FUNCTION):
                    with observe(name="level3", kind=ObserveKind.FUNCTION):
                        with observe(name="level4", kind=ObserveKind.FUNCTION):
                            with observe(name="level5", kind=ObserveKind.FUNCTION):
                                pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 6, f"Expected 6 spans, got {len(spans)}"

        # All spans should have same should_evaluate value
        should_evaluate_values = [span.attributes.get(BasaltSpan.SHOULD_EVALUATE) for span in spans]
        assert all(v is True for v in should_evaluate_values), (
            f"All spans should have should_evaluate=True, got: {should_evaluate_values}"
        )

    def test_multiple_child_branches_propagate(self, setup_tracer):
        """Test propagation across multiple child branches."""
        exporter = setup_tracer

        with start_observe(
            name="root", feature_slug="test", evaluate_config=EvaluationConfig(sample_rate=0.0)
        ):
            # Branch 1
            with observe(name="branch1", kind=ObserveKind.FUNCTION):
                with observe(name="branch1_child", kind=ObserveKind.FUNCTION):
                    pass

            # Branch 2
            with observe(name="branch2", kind=ObserveKind.GENERATION):
                with observe(name="branch2_child", kind=ObserveKind.RETRIEVAL):
                    pass

            # Branch 3
            with observe(name="branch3", kind=ObserveKind.TOOL):
                pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 6, f"Expected 6 spans, got {len(spans)}"

        # All should be False
        for span in spans:
            assert span.attributes.get(BasaltSpan.SHOULD_EVALUATE) is False, (
                f"Span {span.name} should have should_evaluate=False"
            )

    def test_decorator_style_propagation(self, setup_tracer):
        """Test propagation works with decorator-style usage."""
        exporter = setup_tracer

        @start_observe(
            name="decorated_root",
            feature_slug="test",
            evaluate_config=EvaluationConfig(sample_rate=1.0),
        )
        def root_function():
            with observe(name="child_in_decorator", kind=ObserveKind.FUNCTION):
                nested_function()

        @observe(name="nested_decorated", kind=ObserveKind.FUNCTION)
        def nested_function():
            pass

        root_function()

        spans = exporter.get_finished_spans()
        assert len(spans) == 3, f"Expected 3 spans, got {len(spans)}"

        # All should have should_evaluate=True
        for span in spans:
            assert span.attributes.get(BasaltSpan.SHOULD_EVALUATE) is True, (
                f"Span {span.name} should have should_evaluate=True"
            )

    def test_experiment_with_decorator_propagation(self, setup_tracer):
        """Test experiment forces evaluation with decorator pattern."""
        exporter = setup_tracer

        @start_observe(
            name="experiment_decorated",
            feature_slug="test",
            experiment="exp_789",
            evaluate_config=EvaluationConfig(sample_rate=0.0),
        )
        def experiment_function():
            with observe(name="child", kind=ObserveKind.FUNCTION):
                pass

        experiment_function()

        spans = exporter.get_finished_spans()
        assert len(spans) == 2, f"Expected 2 spans, got {len(spans)}"

        # Both should be True due to experiment
        for span in spans:
            assert span.attributes.get(BasaltSpan.SHOULD_EVALUATE) is True, (
                f"Span {span.name} should have should_evaluate=True with experiment"
            )

    def test_mixed_span_kinds_propagation(self, setup_tracer):
        """Test propagation across different span kinds."""
        exporter = setup_tracer

        with start_observe(
            name="mixed_trace",
            feature_slug="test",
            evaluate_config=EvaluationConfig(sample_rate=1.0),
        ):
            with observe(name="generation", kind=ObserveKind.GENERATION):
                pass

            with observe(name="retrieval", kind=ObserveKind.RETRIEVAL):
                pass

            with observe(name="tool", kind=ObserveKind.TOOL):
                pass

            with observe(name="function", kind=ObserveKind.FUNCTION):
                pass

            with observe(name="event", kind=ObserveKind.EVENT):
                pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 6, f"Expected 6 spans, got {len(spans)}"

        # All different kinds should have same should_evaluate
        for span in spans:
            assert span.attributes.get(BasaltSpan.SHOULD_EVALUATE) is True, (
                f"Span {span.name} (kind={span.attributes.get('basalt.span.kind')}) should have should_evaluate=True"
            )

    def test_no_evaluate_config_uses_global_default(self, setup_tracer):
        """Test that without evaluate_config, global default (0.0) is used for all spans."""
        exporter = setup_tracer

        with start_observe(
            name="default_trace",
            feature_slug="test",
            # No evaluate_config, should use global default 0.0
        ):
            with observe(name="child1", kind=ObserveKind.FUNCTION):
                pass

            with observe(name="child2", kind=ObserveKind.GENERATION):
                pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 3, f"Expected 3 spans, got {len(spans)}"

        # All should be False (global default is 0.0)
        for span in spans:
            assert span.attributes.get(BasaltSpan.SHOULD_EVALUATE) is False, (
                f"Span {span.name} should have should_evaluate=False with global default"
            )

    def test_trace_consistency(self, setup_tracer):
        """Test that all spans in a trace have the SAME should_evaluate value."""
        exporter = setup_tracer

        # Test with sample_rate=1.0
        with start_observe(
            name="consistent_trace",
            feature_slug="test",
            evaluate_config=EvaluationConfig(sample_rate=1.0),
        ):
            with observe(name="child1", kind=ObserveKind.FUNCTION):
                with observe(name="grandchild", kind=ObserveKind.GENERATION):
                    pass

        spans = exporter.get_finished_spans()
        should_evaluate_values = [span.attributes.get(BasaltSpan.SHOULD_EVALUATE) for span in spans]

        # All values should be identical
        assert len(set(should_evaluate_values)) == 1, (
            f"All spans should have same should_evaluate value, got: {should_evaluate_values}"
        )
        assert should_evaluate_values[0] is True


class TestExperimentShouldEvaluate:
    """Tests specific to experiment forcing evaluation."""

    def test_experiment_string_forces_evaluation(self, setup_tracer):
        """Test that string experiment ID forces evaluation."""
        exporter = setup_tracer

        with start_observe(
            name="trace",
            feature_slug="test",
            experiment="exp_string",
            evaluate_config=EvaluationConfig(sample_rate=0.0),
        ):
            with observe(name="child", kind=ObserveKind.FUNCTION):
                pass

        spans = exporter.get_finished_spans()

        for span in spans:
            assert span.attributes.get(BasaltSpan.SHOULD_EVALUATE) is True
            # Verify experiment ID is attached to root span
            if span.name == "trace":
                assert "basalt.experiment.id" in span.attributes
                assert span.attributes["basalt.experiment.id"] == "exp_string"

    def test_experiment_object_forces_evaluation(self, setup_tracer):
        """Test that experiment object forces evaluation."""
        exporter = setup_tracer

        class MockExperiment:
            def __init__(self, id, name=None) -> None:
                self.id = id
                self.name = name

        exp = MockExperiment(id="exp_obj", name="Test Experiment")

        with start_observe(
            name="trace",
            feature_slug="test",
            experiment=exp,
            evaluate_config=EvaluationConfig(sample_rate=0.0),
        ):
            with observe(name="child", kind=ObserveKind.FUNCTION):
                pass

        spans = exporter.get_finished_spans()

        for span in spans:
            assert span.attributes.get(BasaltSpan.SHOULD_EVALUATE) is True

    def test_experiment_without_evaluate_config(self, setup_tracer):
        """Test experiment works without explicit evaluate_config."""
        exporter = setup_tracer

        with start_observe(
            name="trace",
            feature_slug="test",
            experiment="exp_no_config",
            # No evaluate_config at all
        ):
            with observe(name="child", kind=ObserveKind.FUNCTION):
                pass

        spans = exporter.get_finished_spans()

        # Should all be True due to experiment
        for span in spans:
            assert span.attributes.get(BasaltSpan.SHOULD_EVALUATE) is True

    def test_no_experiment_respects_sample_rate_zero(self, setup_tracer):
        """Test that without experiment, sample_rate=0.0 is respected."""
        exporter = setup_tracer

        with start_observe(
            name="trace",
            feature_slug="test",
            evaluate_config=EvaluationConfig(sample_rate=0.0),
            # No experiment
        ):
            with observe(name="child", kind=ObserveKind.FUNCTION):
                pass

        spans = exporter.get_finished_spans()

        # Should all be False without experiment
        for span in spans:
            assert span.attributes.get(BasaltSpan.SHOULD_EVALUATE) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
