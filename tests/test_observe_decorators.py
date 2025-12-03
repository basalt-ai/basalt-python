"""Quick test to verify the new observe decorators work correctly."""

import sys


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
        )

        # Test that they are callable
        assert callable(observe)

        # Test that observe accepts kind parameter
        import inspect
        sig = inspect.signature(observe)
        assert "kind" in sig.parameters

        return True
    except (ImportError, AssertionError):
        return False



def test_basic_usage():
    """Test basic decorator usage without actual execution."""
    try:
        from basalt.observability import ObserveKind, observe

        # Test using observe with enum
        @observe(kind=ObserveKind.SPAN, name="test span")
        def test_func1():
            return "test"

        # Test using observe with string
        @observe("test generation", kind=ObserveKind.GENERATION)
        def test_func2():
            return "test"

        # Verify functions are wrapped correctly
        assert callable(test_func1)
        assert callable(test_func2)
        assert test_func1.__name__ == "test_func1"
        assert test_func2.__name__ == "test_func2"

        return True
    except Exception:
        return False


def main():
    """Run all tests."""

    tests = [
        test_imports,
        test_observe_kind_enum,
        test_decorator_definitions,
        test_basic_usage,
    ]

    results = [test() for test in tests]


    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
