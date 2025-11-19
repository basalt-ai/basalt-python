# Getting Started

This guide will help you get started with the Basalt Python SDK.

## Installation

Install the SDK using pip:

```bash
pip install basalt-python
```

## Configuration

Set your Basalt API key as an environment variable:

```bash
export BASALT_API_KEY="your_api_key_here"
```

## Initialization

Initialize the Basalt client with optional global metadata that will be attached to all traces.

```python
from basalt import Basalt, TelemetryConfig

# Optional: Configure telemetry explicitly
telemetry = TelemetryConfig(
    service_name="my-app",
    environment="production"
)

client = Basalt(
    api_key="your_api_key",
    telemetry_config=telemetry,
    observability_metadata={
        "version": "1.0.0",
        "region": "us-east-1"
    }
)
```

## Basic Usage

The Basalt SDK provides a unified `observe` API to track your application's execution. You can use it as a decorator or a context manager.

### Using the Decorator

The simplest way to instrument your code is using the `@observe` decorator.

```python
from basalt.observability import observe

@observe(name="my_function")
def my_function(arg):
    return f"Processed {arg}"

result = my_function("data")
```

### Using the Context Manager

For more granular control, or to trace specific blocks of code, use the `with observe(...)` context manager.

```python
from basalt.observability import observe

def process_data(data):
    with observe(name="process_block") as span:
        # Do some work
        result = data.upper()
        
        # Add metadata
        observe.metadata({"complexity": "low"})
        
        return result
```

## Observability Features

The `observe` API provides static methods to enrich your traces with domain-specific information.

### Identity

Track which user or organization triggered the operation:

```python
@observe(name="chat_handler")
def handle_chat(user_id, message):
    observe.identify(user=user_id, organization="acme-corp")
    # ...
```

### Metadata

Add custom key-value pairs to your spans:

```python
observe.metadata({"model": "gpt-4", "temperature": 0.7})
```

### Input & Output

Explicitly capture input and output data if it's not automatically captured (e.g., inside a context manager):

```python
with observe(name="calculation"):
    data = get_input()
    observe.input(data)
    
    result = perform_calc(data)
    observe.output(result)
```

### Status & Errors

Record success or failure status:

```python
try:
    risky_operation()
    observe.status("ok")
except Exception as e:
    observe.fail(e)
    # or
    observe.status("error", message=str(e))
    raise
```
