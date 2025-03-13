# Basalt Monitor SDK Objects

This directory contains the object classes used by the Basalt Monitor SDK for tracing and monitoring.

## Overview

The Monitor SDK provides a way to trace and monitor the execution of AI workflows. It allows you to create traces, logs, and generations to track the execution of your code.

## Classes

### BaseLog

Base class for logs and generations. It provides common functionality for both logs and generations.

### Log

Class representing a log in the monitoring system. Logs can be used to track the execution of code and can contain generations.

### Generation

Class representing a generation in the monitoring system. Generations are used to track the execution of AI models.

### Trace

Class representing a trace in the monitoring system. Traces are the top-level objects that contain logs and generations.

## Usage

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