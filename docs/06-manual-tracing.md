# Manual Tracing

Learn how to create custom traces using decorators and context managers.

## Overview

Basalt provides two patterns for manual tracing:
- **Decorators**: Apply to functions for automatic span creation
- **Context Managers**: Fine-grained control within code blocks

## Quick Start

### Using Decorators

```python
from basalt.observability import observe_span

@observe_span(name="my.operation")
def process_data(data):
    # Function automatically wrapped in a span
    return transform(data)
```

### Using Context Managers

```python
from basalt.observability import trace_span

def process_request():
    with trace_span(name="request.process") as span:
        span.set_input({"data": "..."})
        result = do_work()
        span.set_output(result)
        return result
```

## Decorators

### @observe_span

Generic span for any operation.

```python
from basalt.observability import observe_span

@observe_span(name="compute.process", attributes={"version": "v2"})
def process(data):
    return result
```

### Nested Spans with Decorators

```python
from basalt.observability import observe_span

@observe_span(name="workflow.main")
def main_workflow():
    step1()  # Creates child span
    step2()  # Creates child span

@observe_span(name="workflow.step1")
def step1():
    pass

@observe_span(name="workflow.step2")
def step2():
    pass
```

## Context Managers

### Basic Context Manager

```python
from basalt.observability import trace_span

with trace_span(name="operation") as span:
    span.set_attribute("key", "value")
    span.set_input({"data": "..."})
    result = do_work()
    span.set_output(result)
```

### SpanHandle Methods

- `set_attribute(key, value)` - Set single attribute
- `set_attributes(dict)` - Set multiple attributes
- `set_input(data)` - Set input data
- `set_output(data)` - Set output data
- `add_event(name, attrs)` - Add timestamped event
- `record_exception(exception)` - Record exception
- `set_status_ok(desc)` - Mark as successful
- `set_status_error(desc)` - Mark as failed

## Span Utilities

Lightweight helpers for working with the current span:

```python
from basalt.observability import (
    get_current_span,
    set_span_attribute,
    add_span_event,
    record_span_exception
)

def my_function():
    # Set attribute on current span
    set_span_attribute("custom.key", "value")

    # Add event
    add_span_event("checkpoint", {"step": 1})

    # Handle exceptions
    try:
        risky_operation()
    except Exception as e:
        record_span_exception(e)
        raise
```