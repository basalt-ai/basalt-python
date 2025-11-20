# Manual Tracing

Learn how to create custom traces using the unified `observe` interface.

## Overview

Basalt provides a unified `observe` API that works as both:
- **Decorator**: Apply to functions for automatic span creation
- **Context Manager**: Fine-grained control within code blocks

## Quick Start

### Using as a Decorator

```python
from basalt.observability import observe

@observe(name="Data Processing")
def process_data(data):
    # Function automatically wrapped in a span
    return transform(data)
```

### Using as a Context Manager

```python
from basalt.observability import observe

def process_request():
    with observe(name="Request Processing") as span:
        span.set_input({"data": "..."})
        result = do_work()
        span.set_output(result)
        return result
```

## The observe Interface

### Basic Usage

`observe` can be used with or without a specific kind. By default, it creates a generic span:

```python
from basalt.observability import observe

@observe(name="Custom Operation", metadata={"version": "v2"})
def my_operation(data):
    return result
```

### Span Kinds

Use the `kind` parameter to specify the type of operation:

```python
from basalt.observability import observe, ObserveKind

# Tool call
@observe(kind=ObserveKind.TOOL, name="Search Database")
def search_db(query):
    return results

# Function call
@observe(kind=ObserveKind.FUNCTION, name="Calculate Total")
def calculate(items):
    return sum(items)

# Event
@observe(kind=ObserveKind.EVENT, name="User Login")
def log_user_login(user_id):
    pass
```

Or use string values:

```python
@observe(kind="tool", name="Search Database")
def search_db(query):
    return results
```

### Nested Spans

Decorators automatically create parent-child relationships:

```python
from basalt.observability import observe

@observe(name="Main Workflow")
def main_workflow():
    prepare_data()  # Creates child span
    process_data()  # Creates child span

@observe(name="Data Preparation")
def prepare_data():
    pass

@observe(name="Data Processing")
def process_data():
    pass
```

## Context Manager Usage

### Working with Span Handles

When using `observe` as a context manager, you get a span handle with helpful methods:

```python
from basalt.observability import observe

with observe(name="API Request") as span:
    # Add metadata
    span.set_attribute("request_id", "abc123")
    
    # Set input
    span.set_input({"endpoint": "/api/users", "method": "GET"})
    
    # Do work
    result = fetch_data()
    
    # Set output
    span.set_output(result)
```

### Span Handle Methods

**Data Methods:**
- `set_input(data)` - Set input payload
- `set_output(data)` - Set output payload
- `set_attribute(key, value)` - Add metadata

**Status Methods:**
- `set_status(status, message)` - Set span status ("ok", "error", or "unset")
- `record_exception(exception)` - Record an exception

**Event Methods:**
- `add_event(name, attributes)` - Add a timestamped event

**Event Methods:**
- `add_event(name, attributes)` - Add a timestamped event

**Example with Error Handling:**

```python
from basalt.observability import observe

with observe(name="Database Query") as span:
    span.set_input({"query": "SELECT * FROM users"})
    
    try:
        result = execute_query()
        span.set_output(result)
        span.set_status("ok")
    except Exception as e:
        span.record_exception(e)
        span.set_status("error", str(e))
        raise
```

## Static Methods

The `observe` class provides static methods for working with the current span:

### Identity Management

```python
from basalt.observability import observe

def handle_request(user_id, org_id):
    # Set user and organization for the current trace
    observe.identify(
        user={"id": user_id, "name": "Alice"},
        organization={"id": org_id, "name": "Acme Corp"}
    )
    
    # Or with just IDs
    observe.identify(user=user_id, organization=org_id)
```

### Metadata Management

```python
from basalt.observability import observe

def process_data():
    # Add metadata to current span
    observe.metadata({"region": "us-west", "version": "v2"})
    
    # Or using kwargs
    observe.metadata(region="us-west", version="v2")
    
    # Merge metadata (updates existing keys)
    observe.update_metadata({"status": "processing"})
```

### Status Management

```python
from basalt.observability import observe

def risky_operation():
    try:
        result = do_work()
        observe.status("ok", "Operation completed successfully")
        return result
    except ValueError as e:
        observe.fail(e)  # Records exception and sets error status
        raise
```

### Accessing the Root Span

Sometimes you need to set metadata on the root span from a nested context:

```python
from basalt.observability import observe

@observe(name="Main Workflow")
def main_workflow():
    # This is the root span
    process_step_1()
    
@observe(name="Step 1")
def process_step_1():
    # From here, we can access the root span
    root = observe.root_span()
    if root:
        root.set_attribute("workflow_status", "processing")
```

## Best Practices

### Set Input and Output

Always set input and output for better observability:

```python
@observe(name="Calculate Discount")
def calculate_discount(user_tier, amount):
    with observe(name="Discount Calculation") as span:
        span.set_input({"user_tier": user_tier, "amount": amount})
        
        discount = compute_discount(user_tier, amount)
        
        span.set_output({"discount": discount})
        return discount
```

### Handle Errors Gracefully

Record exceptions for debugging:

```python
@observe(name="API Call")
def call_external_api():
    with observe(name="External Request") as span:
        try:
            response = make_request()
            span.set_status("ok")
            return response
        except Exception as e:
            span.record_exception(e)
            span.set_status("error", f"API call failed: {str(e)}")
            raise
```