# Getting Started

This guide will help you get started with the Basalt Python SDK.

## Installation

Install the SDK using pip:

```bash
pip install basalt-sdk
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

The Basalt SDK provides `start_observe` and `observe` APIs to track your application's execution.

### Root Span with start_observe

Every trace must begin with a **root span** using `start_observe`. This is the entry point that supports identity tracking, experiment attachment, and evaluator configuration.

```python
from basalt.observability import start_observe, observe


@start_observe(
    feature_slug="workflow",
    name="my_workflow",
    identity={
        "organization": {"id": "123", "name": "ACME"},
        "user": {"id": "456", "name": "John Doe"}
    },
    metadata={"version": "1.0", "environment": "production"}
)
def my_workflow(arg):
    # Identity and metadata automatically propagate to child spans
    observe.set_input({"arg": arg})
    result = process_data(arg)
    observe.set_output({"result": result})
    return result


@observe(name="process_data")
def process_data(data):
    # This is a child span - inherits identity from root
    return f"Processed {data}"


result = my_workflow("data")
```

### Context Manager Form

For more granular control, use context managers:

```python
from basalt.observability import start_observe, observe


def process_request(user_id, data):
    # Create root span with identity
    with start_observe(
            feature_slug="request-processing",
            name="process_request",
            identity={"user": user_id},
            metadata={"source": "api"}
    ):
        observe.set_input({"data": data})

        # Nested span for specific operation
        with observe(name="validate_data", kind="function") as span:
            validated = validate(data)
            span.set_output({"valid": validated})

        result = process(validated)
        observe.set_output({"result": result})
        return result
```

## Observability Features

The `start_observe` and `observe` APIs provide comprehensive ways to enrich your traces.

### Identity Tracking

Track users and organizations at the root span level or dynamically:

```python
from basalt.observability import start_observe, observe


# Method 1: Set on root span (recommended)
@start_observe(
    feature_slug="chat-handler",
    name="chat_handler",
    identity={
        "organization": {"id": "123", "name": "ACME"},
        "user": {"id": "456", "name": "John Doe"}
    }
)
def handle_chat(message):
    # Identity automatically propagates to all child spans
    pass


# Method 2: Set dynamically
@start_observe(feature_slug="api-handler", name="api_handler")
def handle_request(auth_token):
    user_data = verify_token(auth_token)  # Returns {"id": "user-123", "name": "John Doe"}
    observe.set_identity({"user": user_data})
    # Identity now set for entire trace
```

### Metadata

Add custom key-value pairs to your spans:

```python
# On root span
@start_observe(feature_slug="workflow", name="workflow", metadata={"model": "gpt-4", "temperature": 0.7})
def workflow():
    pass

# Or dynamically
observe.metadata({"status": "processing", "batch_id": "123"})
```

### Input & Output

Explicitly capture input and output data if it's not automatically captured (e.g., inside a context manager):

```python
with observe(name="calculation"):
    data = get_input()
    observe.set_input(data)

    result = perform_calc(data)
    observe.set_output(result)
```
