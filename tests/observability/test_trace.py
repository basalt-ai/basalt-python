"""Tests for the trace.identify() method."""

from opentelemetry import trace as otel_trace

from basalt.observability import semconv
from basalt.observability.trace import trace_api as trace

from .utils import get_exporter

# ============================================================================
# Basic Identity Setting Tests
# ============================================================================


def test_identify_user_with_string_id():
    """Test setting user identity with a string ID."""
    exporter = get_exporter()
    exporter.clear()

    tracer = otel_trace.get_tracer(__name__)
    with tracer.start_as_current_span("test_span"):
        trace.identify(user="user-123")

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes[semconv.BasaltUser.ID] == "user-123"
    assert semconv.BasaltUser.NAME not in span.attributes


def test_identify_user_with_dict():
    """Test setting user identity with dict containing id and name."""
    exporter = get_exporter()
    exporter.clear()

    tracer = otel_trace.get_tracer(__name__)
    with tracer.start_as_current_span("test_span"):
        trace.identify(user={"id": "user-123", "name": "Alice"})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes[semconv.BasaltUser.ID] == "user-123"
    assert span.attributes[semconv.BasaltUser.NAME] == "Alice"


def test_identify_organization():
    """Test setting organization identity."""
    exporter = get_exporter()
    exporter.clear()

    tracer = otel_trace.get_tracer(__name__)
    with tracer.start_as_current_span("test_span"):
        trace.identify(organization={"id": "org-456", "name": "ACME Corp"})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes[semconv.BasaltOrganization.ID] == "org-456"
    assert span.attributes[semconv.BasaltOrganization.NAME] == "ACME Corp"


def test_identify_both_user_and_org():
    """Test setting both user and organization identity simultaneously."""
    exporter = get_exporter()
    exporter.clear()

    tracer = otel_trace.get_tracer(__name__)
    with tracer.start_as_current_span("test_span"):
        trace.identify(
            user={"id": "user-123", "name": "Alice"},
            organization={"id": "org-456", "name": "ACME Corp"},
        )

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes[semconv.BasaltUser.ID] == "user-123"
    assert span.attributes[semconv.BasaltUser.NAME] == "Alice"
    assert span.attributes[semconv.BasaltOrganization.ID] == "org-456"
    assert span.attributes[semconv.BasaltOrganization.NAME] == "ACME Corp"


# ============================================================================
# Merge Behavior Tests (Critical)
# ============================================================================


def test_merge_user_name_preserves_id():
    """Test merging user name preserves existing ID."""
    exporter = get_exporter()
    exporter.clear()

    tracer = otel_trace.get_tracer(__name__)
    with tracer.start_as_current_span("test_span"):
        # Set initial identity
        trace.identify(user={"id": "user-123", "name": "Alice"})
        # Merge only name
        trace.identify(user={"name": "Alice Smith"})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    # Verify ID preserved and name updated
    assert span.attributes[semconv.BasaltUser.ID] == "user-123"
    assert span.attributes[semconv.BasaltUser.NAME] == "Alice Smith"


def test_merge_user_id_preserves_name():
    """Test merging user ID preserves existing name."""
    exporter = get_exporter()
    exporter.clear()

    tracer = otel_trace.get_tracer(__name__)
    with tracer.start_as_current_span("test_span"):
        # Set initial identity
        trace.identify(user={"id": "user-123", "name": "Alice"})
        # Merge only ID
        trace.identify(user={"id": "user-456"})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    # Verify name preserved and ID updated
    assert span.attributes[semconv.BasaltUser.ID] == "user-456"
    assert span.attributes[semconv.BasaltUser.NAME] == "Alice"


def test_merge_organization_fields():
    """Test merging organization fields works correctly."""
    exporter = get_exporter()
    exporter.clear()

    tracer = otel_trace.get_tracer(__name__)
    with tracer.start_as_current_span("test_span"):
        # Set initial organization
        trace.identify(organization={"id": "org-123", "name": "ACME Corp"})
        # Merge only name
        trace.identify(organization={"name": "ACME Corporation"})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    # Verify ID preserved and name updated
    assert span.attributes[semconv.BasaltOrganization.ID] == "org-123"
    assert span.attributes[semconv.BasaltOrganization.NAME] == "ACME Corporation"


def test_sequential_merges():
    """Test multiple sequential merges with different fields."""
    exporter = get_exporter()
    exporter.clear()

    tracer = otel_trace.get_tracer(__name__)
    with tracer.start_as_current_span("test_span"):
        # First call: set user ID
        trace.identify(user={"id": "user-123"})
        # Second call: add user name
        trace.identify(user={"name": "Alice"})
        # Third call: update name again
        trace.identify(user={"name": "Alice Smith"})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes[semconv.BasaltUser.ID] == "user-123"
    assert span.attributes[semconv.BasaltUser.NAME] == "Alice Smith"


def test_override_both_fields():
    """Test setting both ID and name twice (complete override)."""
    exporter = get_exporter()
    exporter.clear()

    tracer = otel_trace.get_tracer(__name__)
    with tracer.start_as_current_span("test_span"):
        # First identity
        trace.identify(user={"id": "user-123", "name": "Alice"})
        # Complete override
        trace.identify(user={"id": "user-456", "name": "Bob"})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes[semconv.BasaltUser.ID] == "user-456"
    assert span.attributes[semconv.BasaltUser.NAME] == "Bob"


def test_interleaved_user_org_updates():
    """Test alternating between user and organization updates."""
    exporter = get_exporter()
    exporter.clear()

    tracer = otel_trace.get_tracer(__name__)
    with tracer.start_as_current_span("test_span"):
        # Set user
        trace.identify(user={"id": "user-123"})
        # Set organization
        trace.identify(organization={"id": "org-456"})
        # Update user name
        trace.identify(user={"name": "Alice"})
        # Update organization name
        trace.identify(organization={"name": "ACME Corp"})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    # Verify all fields are present
    assert span.attributes[semconv.BasaltUser.ID] == "user-123"
    assert span.attributes[semconv.BasaltUser.NAME] == "Alice"
    assert span.attributes[semconv.BasaltOrganization.ID] == "org-456"
    assert span.attributes[semconv.BasaltOrganization.NAME] == "ACME Corp"


# ============================================================================
# Edge Case Tests
# ============================================================================


def test_identify_no_current_span():
    """Test that identify is a no-op when there's no current span."""
    # This should not raise an exception
    trace.identify(user={"id": "user-123", "name": "Alice"})
    trace.identify(organization="org-456")


def test_identify_with_none_inputs():
    """Test that None parameters don't modify anything."""
    exporter = get_exporter()
    exporter.clear()

    tracer = otel_trace.get_tracer(__name__)
    with tracer.start_as_current_span("test_span"):
        # Set initial identity
        trace.identify(user={"id": "user-123", "name": "Alice"})
        # Call with None (should not modify)
        trace.identify(user=None, organization=None)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    # Verify nothing changed
    assert span.attributes[semconv.BasaltUser.ID] == "user-123"
    assert span.attributes[semconv.BasaltUser.NAME] == "Alice"
    assert semconv.BasaltOrganization.ID not in span.attributes


def test_identify_with_empty_dict():
    """Test that empty dict doesn't modify anything."""
    exporter = get_exporter()
    exporter.clear()

    tracer = otel_trace.get_tracer(__name__)
    with tracer.start_as_current_span("test_span"):
        # Set initial identity
        trace.identify(user={"id": "user-123", "name": "Alice"})
        # Call with empty dict (should not modify)
        trace.identify(user={})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    # Verify nothing changed
    assert span.attributes[semconv.BasaltUser.ID] == "user-123"
    assert span.attributes[semconv.BasaltUser.NAME] == "Alice"


def test_identify_partial_dict():
    """Test dict with only id or only name."""
    exporter = get_exporter()
    exporter.clear()

    tracer = otel_trace.get_tracer(__name__)
    with tracer.start_as_current_span("test_span"):
        # Set only ID
        trace.identify(user={"id": "user-123"})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes[semconv.BasaltUser.ID] == "user-123"
    assert semconv.BasaltUser.NAME not in span.attributes

    # Start new span and add only name
    exporter.clear()
    with tracer.start_as_current_span("test_span2"):
        trace.identify(user={"id": "user-123"})
        trace.identify(user={"name": "Alice"})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes[semconv.BasaltUser.ID] == "user-123"
    assert span.attributes[semconv.BasaltUser.NAME] == "Alice"


def test_type_coercion():
    """Test that numeric IDs are converted to strings."""
    exporter = get_exporter()
    exporter.clear()

    tracer = otel_trace.get_tracer(__name__)
    with tracer.start_as_current_span("test_span"):
        # Pass numeric ID
        trace.identify(user={"id": 12345, "name": "Alice"})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes[semconv.BasaltUser.ID] == "12345"
    assert span.attributes[semconv.BasaltUser.NAME] == "Alice"


def test_identify_with_empty_string():
    """Test that empty string values set attributes to empty."""
    exporter = get_exporter()
    exporter.clear()

    tracer = otel_trace.get_tracer(__name__)
    with tracer.start_as_current_span("test_span"):
        # Set initial identity
        trace.identify(user={"id": "user-123", "name": "Alice"})
        # Set name to empty string
        trace.identify(user={"name": ""})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes[semconv.BasaltUser.ID] == "user-123"
    assert span.attributes[semconv.BasaltUser.NAME] == ""


def test_identify_with_none_dict_value():
    """Test that None values in dict set attributes to empty."""
    exporter = get_exporter()
    exporter.clear()

    tracer = otel_trace.get_tracer(__name__)
    with tracer.start_as_current_span("test_span"):
        # Set initial identity
        trace.identify(user={"id": "user-123", "name": "Alice"})
        # Set name to None (should become empty string)
        trace.identify(user={"name": None})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes[semconv.BasaltUser.ID] == "user-123"
    assert span.attributes[semconv.BasaltUser.NAME] == ""


# ============================================================================
# Integration Tests
# ============================================================================


def test_identify_with_other_trace_methods():
    """Test that identify works alongside other trace methods."""
    exporter = get_exporter()
    exporter.clear()

    tracer = otel_trace.get_tracer(__name__)
    with tracer.start_as_current_span("test_span"):
        # Use identify
        trace.identify(user={"id": "user-123", "name": "Alice"})
        # Use other trace methods
        trace.set_attribute("custom.key", "value")
        trace.add_event("test_event", {"event_key": "event_value"})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    # Verify all attributes are present
    assert span.attributes[semconv.BasaltUser.ID] == "user-123"
    assert span.attributes[semconv.BasaltUser.NAME] == "Alice"
    assert span.attributes["custom.key"] == "value"


def test_identify_in_context_manager():
    """Test identify within start_as_current_span context manager."""
    exporter = get_exporter()
    exporter.clear()

    tracer = otel_trace.get_tracer(__name__)
    with tracer.start_as_current_span("test_span"):
        trace.identify(user="user-123")
        trace.identify(organization={"id": "org-456", "name": "ACME"})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes[semconv.BasaltUser.ID] == "user-123"
    assert span.attributes[semconv.BasaltOrganization.ID] == "org-456"
    assert span.attributes[semconv.BasaltOrganization.NAME] == "ACME"


def test_identify_persistence():
    """Test that identity persists through span lifecycle."""
    exporter = get_exporter()
    exporter.clear()

    tracer = otel_trace.get_tracer(__name__)
    with tracer.start_as_current_span("test_span"):
        # Set identity at the start
        trace.identify(user={"id": "user-123", "name": "Alice"})
        # Do some work
        trace.set_attribute("step", "1")
        # Update identity
        trace.identify(user={"name": "Alice Smith"})
        # Do more work
        trace.set_attribute("step", "2")

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    # Verify final state
    assert span.attributes[semconv.BasaltUser.ID] == "user-123"
    assert span.attributes[semconv.BasaltUser.NAME] == "Alice Smith"
    assert span.attributes["step"] == "2"
