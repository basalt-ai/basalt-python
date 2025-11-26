# Manual Tracing

Learn how to create custom traces using the unified `start_observe` and `observe` interfaces.

## Overview

Basalt provides two main observability APIs:
- **`start_observe`**: Creates root spans with identity, experiment, and evaluator configuration
- **`observe`**: Creates nested spans with different kinds (generation, retrieval, tool, etc.)

Both work as:
- **Decorator**: Apply to functions for automatic span creation
- **Context Manager**: Fine-grained control within code blocks

## Quick Start

### Root Span with start_observe

Every trace must begin with a root span created using `start_observe`:

```python
from basalt.observability import start_observe, observe

@start_observe(
    name="Data Processing Workflow",
    identity={
        "organization": {"id": "123", "name": "ACME"},
        "user": {"id": "456", "name": "John Doe"}
    },
    metadata={"version": "v2", "environment": "production"}
)
def process_data(data):
    # Identity and metadata automatically propagate to child spans
    prepare_data(data)
    transform_data(data)
    return result

@observe(name="Data Preparation")
def prepare_data(data):
    # This is a child span - inherits identity from root
    pass
```

### Using as a Decorator

```python
from basalt.observability import observe

@observe(name="Data Processing", kind="function")
def process_data(data):
    # Function automatically wrapped in a span
    return transform(data)
```

### Using as a Context Manager

```python
from basalt.observability import start_observe


def process_request():
    with start_observe(
            name="Request Processing",
            identity={
                "organization": {"id": "123", "name": "ACME"},
                "user": {"id": "456", "name": "John Doe"}
            }
    ):
        observe.set_input({"data": "..."})
        result = do_work()
        observe.set_output(result)
        return result
```

## The start_observe and observe Interfaces

### start_observe: Root Span Entry Point

Use `start_observe` to create the root span of your trace. This is required as the entry point and supports:

```python
from basalt.observability import start_observe

# With identity tracking
@start_observe(
    name="Main Workflow",
    identity={
        "organization": {"id": "123", "name": "ACME"},
        "user": {"id": "456", "name": "John Doe"}
    },
    metadata={"environment": "production"}
)
def main_workflow(data):
    return process(data)

# With experiment tracking
@start_observe(
    name="A/B Test",
    experiment={"id": "exp_001", "name": "Model Comparison", "variant": "model_a"},
    identity={
        "organization": {"id": "123", "name": "ACME"},
        "user": {"id": "456", "name": "John Doe"}
    }
)
def run_experiment():
    return results

# Context manager form
with start_observe(
    name="Batch Job",
    identity={
        "organization": {"id": "123", "name": "ACME"},
        "user": {"id": "456", "name": "John Doe"}
    }
):
    # Your code here
    pass
```

**Key `start_observe` parameters:**
- `name`: Span name (defaults to function name if used as decorator)
- `identity`: Dict with `user` and/or `organization` keys for tracking
- `experiment`: Dict with `id`, `name`, and `variant` for A/B testing
- `evaluate_config`: Evaluator configuration for the root span
- `metadata`: Custom key-value pairs

### observe: Nested Spans

Use `observe` for nested operations within a root span:

```python
from basalt.observability import observe, ObserveKind

@observe(name="Custom Operation", kind=ObserveKind.FUNCTION, metadata={"version": "v2"})
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

Decorators automatically create parent-child relationships. Always start with `start_observe` as the root:

```python
from basalt.observability import start_observe, observe

@start_observe(
    name="Main Workflow",
    identity={
        "organization": {"id": "123", "name": "ACME"},
        "user": {"id": "456", "name": "John Doe"}
    }
)
def main_workflow():
    # Root span created here
    prepare_data()  # Creates child span
    process_data()  # Creates child span
    return result

@observe(name="Data Preparation")
def prepare_data():
    # Child span - inherits identity from root
    pass

@observe(name="Data Processing")
def process_data():
    # Another child span
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

