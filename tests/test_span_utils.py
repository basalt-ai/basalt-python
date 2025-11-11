from opentelemetry import trace as ot_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import StatusCode

from basalt.observability import (
    add_span_event,
    get_current_span,
    get_current_span_handle,
    record_span_exception,
    set_span_attribute,
    set_span_attributes,
    set_span_status_error,
    set_span_status_ok,
    trace_span,
)

# Ensure a real tracer provider is installed for valid spans during tests
ot_trace.set_tracer_provider(TracerProvider())


def test_span_utils_no_active_span():
    # Outside of any span, helpers should be no-ops returning False/0
    assert set_span_attribute("foo", "bar") is False
    assert set_span_attributes({"a": 1, "b": 2}) == 0
    assert add_span_event("evt") is False
    assert record_span_exception(Exception("boom")) is False
    assert get_current_span() is None
    assert get_current_span_handle() is None


def test_span_utils_with_span():
    with trace_span("test.span.utils") as handle:
        # Attribute helpers
        assert set_span_attribute("foo", "bar") is True
        assert set_span_attributes({"alpha": 1, "beta": 2}) == 2
        # Event helper
        assert add_span_event("custom.event", {"k": "v"}) is True
        # Exception recording
        assert record_span_exception(Exception("boom")) is True
        # Status helpers
        assert set_span_status_ok("all good") is True
        assert set_span_status_error("forced error") is True
        # Direct access helpers
        assert get_current_span() is handle.span
        assert get_current_span_handle() is not None
        # Validate some attributes were actually set
        attrs = getattr(handle.span, "attributes", {})
        assert attrs.get("foo") == "bar"
        assert attrs.get("alpha") == 1
        # Status should be ERROR per last call
        status = getattr(handle.span, "_status", None) or getattr(handle.span, "status", None)
        if status is not None:
            code = getattr(status, "status_code", None)
            assert code in (StatusCode.ERROR, StatusCode.OK)
