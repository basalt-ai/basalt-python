# Basalt SDK

Basalt is a powerful tool for managing AI prompts, monitoring AI applications, and their release workflows. This SDK is the official Python package for interacting with your Basalt prompts and monitoring your AI applications.

## Installation

Install the Basalt SDK via pip:

```bash
pip install basalt-sdk
```

## Usage

### Importing and Initializing the SDK

To get started, import the `Basalt` class and initialize it with your API key. Telemetry is enabled by default via OpenTelemetry, but can be configured or disabled:

```python
from basalt import Basalt, OpenLLMetryConfig, TelemetryConfig

# Basic initialization with API key
basalt = Basalt(api_key="my-dev-api-key")

# Disable all telemetry
basalt = Basalt(api_key="my-dev-api-key", enable_telemetry=False)

# Advanced telemetry configuration
telemetry = TelemetryConfig(
    service_name="my-app",
    environment="staging",
    enable_openllmetry=True,
    openllmetry_config=OpenLLMetryConfig(trace_content=False),
)
basalt = Basalt(api_key="my-dev-api-key", telemetry_config=telemetry)

# Don't forget to shutdown the client when done
# This flushes any pending telemetry data
basalt.shutdown()
```

See `examples/telemetry_example.py` for a more complete walkthrough covering decorators, context managers, and custom exporters.

### Telemetry & Observability

The SDK includes comprehensive OpenTelemetry integration for observability:

- `TelemetryConfig` centralizes all observability options (service name/version, exporter overrides, HTTP auto-instrumentation, etc.).
- `OpenLLMetryConfig` enables Traceloop/OpenLLMetry integrations across all supported LLM providers.
- Quick disable via `enable_telemetry=False` bypasses all instrumentation without touching application code.
- Built-in decorators and context managers simplify manual span creation:
  - **Decorators**: `@trace_operation`, `@trace_llm`, `@trace_http`
  - **Context Managers**: `trace_span()`, `trace_llm_call()`, `trace_retrieval()`

```python
from basalt.observability.decorators import trace_operation, trace_llm
from basalt.observability.context_managers import trace_span, trace_llm_call

# Using decorators for automatic tracing
@trace_operation(name="dataset.process")
def process_dataset(slug: str) -> str:
    return f"processed:{slug}"

@trace_llm(name="llm.generate")
def generate_summary(model: str, prompt: str) -> dict:
    # Your LLM call here
    return {"choices": [{"message": {"content": "Summary"}}]}

# Using context managers for manual tracing
def custom_workflow():
    with trace_span("custom.section", attributes={"feature": "demo"}) as span:
        span.add_event("processing_started")
        # Your code here
        span.set_attribute("status", "completed")

    with trace_llm_call("manual.llm") as llm_span:
        llm_span.set_model("gpt-4")
        llm_span.set_prompt("Tell me a joke")
        # Make your LLM call
        llm_span.set_completion("Response here")
        llm_span.set_tokens(input=10, output=20)
```

**Supported environment variables:**

| Variable | Description |
| --- | --- |
| `BASALT_TELEMETRY_ENABLED` | Master switch to enable/disable telemetry (default: `true`). |
| `BASALT_SERVICE_NAME` | Overrides the OTEL `service.name`. |
| `BASALT_ENVIRONMENT` | Sets `deployment.environment`. |
| `BASALT_OTEL_EXPORTER_OTLP_ENDPOINT` | Custom OTLP HTTP endpoint for traces. |
| `TRACELOOP_TRACE_CONTENT` | Controls whether prompts/completions are logged. |
| `TRACELOOP_TELEMETRY` | Controls OpenLLMetry anonymous telemetry. |

## Prompt SDK

The Prompt SDK allows you to interact with your Basalt prompts using an exception-based API for clear error handling.

For a complete working example, check out:
- [Prompt API Example](./examples/prompt_api_example.py) - Detailed examples with error handling
- [Prompt SDK Demo Notebook](./examples/prompt_sdk_demo.ipynb) - Interactive notebook

### Available Methods

#### Prompts
Your Basalt instance exposes a `prompts` property for interacting with your Basalt prompts:

- **List Prompts**

  Retrieve all available prompts.

  **Example Usage:**

  ```python
  from basalt import Basalt
  from basalt.types.exceptions import BasaltAPIError, UnauthorizedError

  basalt = Basalt(api_key="your-api-key")

  try:
      prompts = basalt.prompts.list_sync()
      for prompt in prompts:
          print(f"{prompt.slug} - {prompt.name}")
  except UnauthorizedError:
      print("Invalid API key")
  except BasaltAPIError as e:
      print(f"API error: {e}")
  ```

- **Get a Prompt**

  Retrieve a specific prompt using a slug, and optional filters `tag` and `version`. Without tag or version, the production version of your prompt is selected by default.

  **Example Usage:**

  ```python
  from basalt import Basalt
  from basalt.types.exceptions import NotFoundError, BasaltAPIError

  basalt = Basalt(api_key="your-api-key")

  try:
      # Get the production version
      prompt = basalt.prompts.get_sync('prompt-slug')
      print(prompt.text)

      # With optional tag or version parameters
      prompt = basalt.prompts.get_sync(slug='prompt-slug', tag='latest')
      prompt = basalt.prompts.get_sync(slug='prompt-slug', version='1.0.0')

      # If your prompt has variables, pass them when fetching
      prompt = basalt.prompts.get_sync(
          slug='prompt-slug',
          variables={'name': 'John Doe', 'role': 'engineer'}
      )

      # Use the prompt with your AI provider of choice
      # Example: OpenAI
      import openai
      client = openai.OpenAI()
      
      response = client.chat.completions.create(
          model='gpt-4',
          messages=[{'role': 'user', 'content': prompt.text}]
      )
      print(response.choices[0].message.content)

  except NotFoundError:
      print('Prompt not found')
  except BasaltAPIError as e:
      print(f'API error: {e}')
  finally:
      basalt.shutdown()
  ```

- **Describe a Prompt**

  Get metadata about a prompt including available versions and tags.

  **Example Usage:**

  ```python
  try:
      description = basalt.prompts.describe_sync('prompt-slug')
      print(f"Available versions: {description.available_versions}")
      print(f"Available tags: {description.available_tags}")
  except NotFoundError:
      print('Prompt not found')
  ```

- **Async Operations**

  All methods have async variants using `_async` suffix:

  ```python
  import asyncio

  async def fetch_prompts():
      basalt = Basalt(api_key="your-api-key")
      
      try:
          # List prompts asynchronously
          prompts = await basalt.prompts.list_async()
          
          # Get a specific prompt asynchronously
          prompt = await basalt.prompts.get_async('prompt-slug')
          
          # Describe a prompt asynchronously
          description = await basalt.prompts.describe_async('prompt-slug')
          
      finally:
          basalt.shutdown()

  asyncio.run(fetch_prompts())
  ```

## Dataset SDK

The Dataset SDK allows you to interact with your Basalt datasets using an exception-based API for clear error handling.

For a complete working example, check out:
- [Dataset API Example](./examples/dataset_api_example.py) - Detailed examples with error handling
- [Dataset SDK Demo Notebook](./examples/dataset_sdk_demo.ipynb) - Interactive notebook

### Available Methods

#### Datasets
Your Basalt instance exposes a `datasets` property for interacting with your Basalt datasets:

- **List Datasets**

  Retrieve all available datasets.

  **Example Usage:**

  ```python
  from basalt import Basalt
  from basalt.types.exceptions import BasaltAPIError

  basalt = Basalt(api_key="your-api-key")

  try:
      datasets = basalt.datasets.list_sync()
      for dataset in datasets:
          print(f"{dataset.slug} - {dataset.name}")
          print(f"Columns: {dataset.columns}")
  except BasaltAPIError as e:
      print(f"API error: {e}")
  ```

- **Get a Dataset**

  Retrieve a specific dataset by slug.

  **Example Usage:**

  ```python
  from basalt.types.exceptions import NotFoundError

  try:
      dataset = basalt.datasets.get_sync('dataset-slug')
      print(f"Dataset: {dataset.name}")
      print(f"Rows: {len(dataset.rows)}")
      
      # Access dataset rows
      for row in dataset.rows:
          print(row)
          
  except NotFoundError:
      print('Dataset not found')
  ```

- **Async Operations**

  All methods have async variants using `_async` suffix:

  ```python
  import asyncio

  async def fetch_datasets():
      basalt = Basalt(api_key="your-api-key")
      
      try:
          # List datasets asynchronously
          datasets = await basalt.datasets.list_async()
          
          # Get a specific dataset asynchronously
          dataset = await basalt.datasets.get_async('dataset-slug')
          
      finally:
          basalt.shutdown()

  asyncio.run(fetch_datasets())
  ```

## Error Handling

The SDK uses exception-based error handling for clarity and pythonic patterns:

```python
from basalt import Basalt
from basalt.types.exceptions import (
    BasaltAPIError,      # Base exception for all API errors
    NotFoundError,       # Resource not found (404)
    UnauthorizedError,   # Authentication failed (401)
    NetworkError,        # Network/connection errors
)

basalt = Basalt(api_key="your-api-key")

try:
    prompt = basalt.prompts.get_sync('my-prompt')
    # Use the prompt
except NotFoundError:
    print("Prompt doesn't exist")
except UnauthorizedError:
    print("Check your API key")
except NetworkError:
    print("Network connection failed")
except BasaltAPIError as e:
    print(f"Other API error: {e}")
finally:
    basalt.shutdown()
```

## License

MIT
