# Basalt SDK

Basalt is a powerful tool for managing AI prompts, monitoring AI applications, and their release workflows. This SDK is the official Python package for interacting with your Basalt prompts and monitoring your AI applications.

## Installation

Install the Basalt SDK via pip:

```bash
pip install basalt-sdk
```

## Usage

### Importing and Initializing the SDK

To get started, import the `Basalt` class and initialize it with your API key:

```python
from basalt import Basalt

# Basic initialization with API key
basalt = Basalt(api_key="my-dev-api-key")

# Specify a log_level
basalt = Basalt(api_key="my-dev-api-key", log_level="debug")

# Or with an environment variable
import os
basalt = Basalt(api_key=os.getenv("BASALT_API_KEY"))
```

## Prompt SDK

The Prompt SDK allows you to interact with your Basalt prompts.

For a complete working example, check out our [Prompt SDK Demo Notebook](./examples/prompt_sdk_demo.ipynb).

### Available Methods

#### Prompts
Your Basalt instance exposes a `prompt` property for interacting with your Basalt prompts:

- **Get a Prompt**

  Retrieve a specific prompt using a slug, and optional filters `tag` and `version`. Without tag or version, the production version of your prompt is selected by default.

  **Example Usage:**

  ```python
  error, result = basalt.prompt.get('prompt-slug')

  # With optional tag or version parameters
  error, result = basalt.prompt.get(slug='prompt-slug', tag='latest')
  error, result = basalt.prompt.get(slug='prompt-slug', version='1.0.0')

  # If your prompt has variables,
  # pass them when fetching your prompt
  error, result = basalt.prompt.get(slug='prompt-slug', variables={ 'name': 'John Doe' })

  # Handle the result by unwrapping the error / value
  if error:
      print('Could not fetch prompt', error)
  else:
      # Use the prompt with your AI provider of choice
      # Example: OpenAI
      openai_client.chat_completion.create(
          model='gpt-4',
          messages=[{'role': 'user', 'content': result.prompt}]
      )
  ```

## Monitor SDK

The Monitor SDK allows you to track and monitor your AI application's execution through traces, logs, and generations.

For a complete working example, check out our [Monitor SDK Demo Notebook](./examples/monitor_sdk_demo.ipynb).

### Creating a Trace

A trace represents a complete execution flow in your application:

```python
# Create a trace
trace = basalt.monitor.create_trace(
    "slug",  # Chain slug - identifies this type of workflow
    {
        "input": "What are the benefits of AI in healthcare?",
        "user": {"id": "user123", "name": "John Doe"},
        "organization": {"id": "org123", "name": "Healthcare Inc"},
        "metadata": {"source": "web", "priority": "high"}
    }
)
```

### Adding Logs to a Trace

Logs represent individual steps or operations within a trace:

```python
# Create a log for content moderation
moderation_log = trace.create_log({
    "type": "span",
    "name": "content-moderation",
    "input": trace.input,
    "metadata": {"model": "text-moderation-latest"},
    "user": {"id": "user123", "name": "John Doe"},
    "organization": {"id": "org123", "name": "Healthcare Inc"}
})

# Update and end the log
moderation_log.update({"metadata": {"completed": True}})
moderation_log.end({"flagged": False, "categories": [], "scores": {}})
```

### Creating and Managing Generations

Generations are special types of logs specifically for AI model interactions:

```python
# Create a log for the main processing
main_log = trace.create_log({
    "type": "span",
    "name": "main-processing",
    "user": {"id": "user123", "name": "John Doe"},
    "organization": {"id": "org123", "name": "Healthcare Inc"},
    "input": trace.input
})

# Create a generation within the main log using a prompt from Basalt
generation = main_log.create_generation({
    "name": "healthcare-benefits-generation",
    "input": trace.input,
    "prompt": {
        "slug": "prompt-slug", # This tells the SDK to fetch the prompt from Basalt
        "version": "0.1" # This specifies the version to use
    }
})

# Or create a generation not managed in Basalt
generation = main_log.create_generation({
    "name": "healthcare-benefits-generation",
    "user": {"id": "user123", "name": "John Doe"},
    "organization": {"id": "org123", "name": "Healthcare Inc"},
    "input": trace.input
})

# End the generation with the response
generation.end("AI generated response")

# End the log and trace
main_log.end("Final output")
trace.end("End of trace")
```

### Complex Workflows with Nested Logs

You can create complex workflows with nested logs and multiple generations:

```python
# Create a nested log
nested_log = parent_log.create_log({
    "type": "span",
    "name": "nested-process",
    "metadata": {"key": "value"},
    "input": parent_log.input
})

# Create generations within nested logs
nested_generation = nested_log.create_generation({
    "name": "nested-generation",
    "input": nested_log.input,
    "prompt": {"slug": "another-prompt", "version": "0.1"},
    "variables": {"variable_example": "test variable"}
})

# End all logs in reverse order
nested_generation.end("Generation output")
nested_log.end("Nested log output")
parent_log.end("Parent log output")
trace.end("End of trace")
```

## License

MIT
