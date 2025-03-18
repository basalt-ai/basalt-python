# Basalt SDK

This directory contains the SDK classes for interacting with Basalt services.

## PromptSDK

The `PromptSDK` class provides methods for interacting with Basalt prompts. It allows you to retrieve, describe, and list prompts.

## MonitorSDK

The `MonitorSDK` class provides methods for monitoring and tracing the execution of AI workflows. It allows you to create traces, logs, and generations to track the execution of your code.

### Methods

#### `create_trace(slug, params=None)`

Creates a new trace for monitoring.

- `slug` (str): The unique identifier for the trace.
- `params` (Optional[Dict[str, Any]]): Optional parameters for the trace.
- Returns: A new `Trace` instance.

#### `create_generation(params)`

Creates a new generation for monitoring.

- `params` (Dict[str, Any]): Parameters for the generation.
- Returns: A new `Generation` instance.

#### `create_log(params)`

Creates a new log for monitoring.

- `params` (Dict[str, Any]): Parameters for the log.
- Returns: A new `Log` instance.

### Usage

```python
from basalt.sdk import Basalt

# Initialize Basalt with your API key
basalt = Basalt(api_key="your-api-key")

# Create a trace
trace = basalt.monitor.create_trace(
    "my-trace",
    {
        "input": "My input",
        "user": {"id": "user123", "name": "John Doe"},
        "organization": {"id": "org123", "name": "Acme Inc."},
        "metadata": {"property1": "value1", "property2": "value2"}
    }
)

# Create a log
log = trace.create_log({
    "name": "my-log",
    "input": "My log input"
})

# Create a generation
generation = log.create_generation({
    "name": "my-generation",
    "input": "My generation input",
    "prompt": {"slug": "my-prompt"},
    "variables": {"var1": "value1"}
})

# Update the generation
generation.update({
    "output": "My generation output",
    "metadata": {"property1": "value1"}
})

# End the generation
generation.end("My generation output")

# End the log
log.end("My log output")

# End the trace
trace.end("My trace output")
```

## Advanced Usage

See the test file `tests/test_monitor_sdk.py` for more advanced usage examples. 