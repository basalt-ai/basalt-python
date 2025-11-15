"""Quick test to verify the new observe decorators work correctly."""

import sys


def test_imports():
    """Test that all new decorators can be imported."""
    try:
        from basalt.observability import (
            ObserveKind,
            observe,
            observe_event,
            observe_function,
            observe_generation,
            observe_retrieval,
            observe_span,
            observe_tool,
        )
        return True
    except ImportError:
        return False


def test_observe_kind_enum():
    """Test ObserveKind enum values."""
    try:
        from basalt.observability import ObserveKind

        assert ObserveKind.SPAN.value == "span"
        assert ObserveKind.GENERATION.value == "generation"
        assert ObserveKind.RETRIEVAL.value == "retrieval"
        assert ObserveKind.FUNCTION.value == "function"
        assert ObserveKind.TOOL.value == "tool"
        assert ObserveKind.EVENT.value == "event"

        return True
    except (ImportError, AssertionError):
        return False


def test_decorator_definitions():
    """Test that decorators are callable and have correct signatures."""
    try:
        from basalt.observability import (
            observe,
            observe_generation,
            observe_span,
        )

        # Test that they are callable
        assert callable(observe)
        assert callable(observe_generation)
        assert callable(observe_span)

        # Test that observe accepts kind parameter
        import inspect
        sig = inspect.signature(observe)
        assert "kind" in sig.parameters

        return True
    except (ImportError, AssertionError):
        return False


def test_backward_compatibility():
    """Test that old trace_* decorators still exist (with deprecation)."""
    try:
        import warnings

        from basalt.observability.decorators import (
            trace_event,
            trace_generation,
            trace_retrieval,
            trace_span,
        )

        # Test that they are callable
        assert callable(trace_generation)
        assert callable(trace_span)
        assert callable(trace_retrieval)
        assert callable(trace_event)

        return True
    except (ImportError, AssertionError):
        return False


def test_basic_usage():
    """Test basic decorator usage without actual execution."""
    try:
        from basalt.observability import ObserveKind, observe, observe_generation

        # Test using observe with enum
        @observe(ObserveKind.SPAN, name="test.span")
        def test_func1():
            return "test"

        # Test using observe with string
        @observe("generation", name="test.gen")
        def test_func2():
            return "test"

        # Test using specialized decorator
        @observe_generation(name="test.specialized")
        def test_func3():
            return "test"

        # Verify functions are wrapped correctly
        assert callable(test_func1)
        assert callable(test_func2)
        assert callable(test_func3)
        assert test_func1.__name__ == "test_func1"
        assert test_func2.__name__ == "test_func2"
        assert test_func3.__name__ == "test_func3"

        return True
    except Exception:
        return False


def main():
    """Run all tests."""

    tests = [
        test_imports,
        test_observe_kind_enum,
        test_decorator_definitions,
        test_backward_compatibility,
        test_basic_usage,
    ]

    results = [test() for test in tests]


    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
