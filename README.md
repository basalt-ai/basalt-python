# Basalt SDK

Basalt is a powerful tool for managing AI prompts, monitoring AI applications, and their release workflows. This SDK is the official Python package for interacting with your Basalt prompts and monitoring your AI applications.

## Installation

Install the Basalt SDK via pip:

```bash
pip install basalt-sdk
```

### Optional Instrumentation Dependencies

The SDK includes optional OpenTelemetry instrumentation packages for various LLM providers, vector databases, and frameworks. You can install only the instrumentations you need:

#### LLM Provider Instrumentations

```bash
# Individual providers (10 available)
pip install basalt-sdk[openai]
pip install basalt-sdk[anthropic]
pip install basalt-sdk[google-generativeai]  # Google Gemini
pip install basalt-sdk[cohere]
pip install basalt-sdk[bedrock]
pip install basalt-sdk[vertex-ai]
pip install basalt-sdk[ollama]
pip install basalt-sdk[mistralai]
pip install basalt-sdk[together]
pip install basalt-sdk[replicate]

# Multiple providers
pip install basalt-sdk[openai,anthropic]

# All LLM providers (10 providers)
pip install basalt-sdk[llm-all]
```

**Note:** The NEW Google GenAI SDK instrumentation is not yet available on PyPI. Use `google-generativeai` for Gemini for now.

#### Vector Database Instrumentations

```bash
# Individual vector databases
pip install basalt-sdk[chromadb]
pip install basalt-sdk[pinecone]
pip install basalt-sdk[qdrant]

# All vector databases
pip install basalt-sdk[vector-all]
```

#### Framework Instrumentations

```bash
# Individual frameworks
pip install basalt-sdk[langchain]
pip install basalt-sdk[llamaindex]
pip install basalt-sdk[haystack]

# All frameworks
pip install basalt-sdk[framework-all]
```

#### Install Everything

```bash
# Install all available instrumentations
pip install basalt-sdk[all]
```

**Note:** These instrumentation packages are automatically activated when you enable telemetry in the Basalt SDK. They provide automatic tracing for your LLM provider calls, vector database operations, and framework usage.

## Usage

### Importing and Initializing the SDK

To get started, import the `Basalt` class and initialize it with your API key. Telemetry is enabled by default via OpenTelemetry, but can be configured or disabled:

```python
from basalt import Basalt, TelemetryConfig

# Basic initialization with API key
basalt = Basalt(api_key="my-dev-api-key")

# Disable all telemetry
basalt = Basalt(api_key="my-dev-api-key", enable_telemetry=False)

# Advanced telemetry configuration
telemetry = TelemetryConfig(
    service_name="my-app",
    environment="staging",
    enable_llm_instrumentation=True,
    llm_trace_content=False,
    llm_enabled_providers=["openai", "anthropic"],  # Optional: selective instrumentation
)
basalt = Basalt(api_key="my-dev-api-key", telemetry_config=telemetry)

# Configure default trace context when constructing the client
basalt = Basalt(
    api_key="my-dev-api-key",
    trace_user={"id": "user-123", "name": "Ada"},
    trace_organization={"id": "org-7", "name": "Acme"},
    trace_experiment={"id": "exp-1", "feature_slug": "agent"},
    trace_evaluators=["accuracy", "toxicity"],
    trace_metadata={"env": "staging"},
)

# Don't forget to shutdown the client when done
# This flushes any pending telemetry data
basalt.shutdown()
```

See `examples/telemetry_example.py` for a more complete walkthrough covering decorators, context managers, and custom exporters.

### Telemetry & Observability

The SDK includes comprehensive OpenTelemetry integration for observability:

- `TelemetryConfig` centralizes all observability options including:
  - Service name/version and deployment environment
  - Custom exporter configuration
  - Lightweight tracing wrappers for Basalt API calls (bring your own HTTP instrumentation if you need transport-level spans)
  - LLM provider instrumentation with fine-grained control over which providers to instrument
- Quick disable via `enable_telemetry=False` bypasses all instrumentation without touching application code.
- Built-in decorators and context managers simplify manual span creation:
  - **Decorators**: `@trace_span`, `@trace_generation`, `@trace_http`
  - **Context Managers**: `trace_span()`, `trace_generation()`, `trace_retrieval()`, `trace_tool()`, `trace_event()`

```python
from basalt.observability import trace_event, trace_generation, trace_span, trace_tool
from basalt.observability.decorators import trace_generation as trace_generation_decorator, trace_span as trace_span_decorator

# Using decorators for automatic tracing
@trace_span_decorator(name="dataset.process")
def process_dataset(slug: str) -> str:
    return f"processed:{slug}"

@trace_generation_decorator(name="llm.generate")
def generate_summary(model: str, prompt: str) -> dict:
    # Your LLM call here
    return {"choices": [{"message": {"content": "Summary"}}]}

# Using context managers for manual tracing
def custom_workflow():
    with trace_span("custom.section", attributes={"feature": "demo"}) as span:
        span.add_event("processing_started")
        # Your code here
        span.set_attribute("status", "completed")

    with trace_generation("manual.llm") as llm_span:
        llm_span.set_model("gpt-4")
        llm_span.set_prompt("Tell me a joke")
        # Make your LLM call
        llm_span.set_completion("Response here")
        llm_span.set_tokens(input=10, output=20)

    with trace_tool("manual.tool") as tool_span:
        tool_span.set_tool_name("http_fetch")
        tool_span.set_input({"url": "https://getbasalt.ai"})
        # Execute your tool call
        tool_span.set_output({"status": 200})

    with trace_event("workflow.event") as event_span:
        event_span.set_event_type("checkpoint")
        event_span.set_payload({"step": 3, "status": "ok"})
```

#### Trace context helpers

- Configure defaults once at startup:

  ```python
  from basalt.observability import (
      add_default_evaluators,
      attach_trace_experiment,
      configure_trace_defaults,
      set_trace_organization,
      set_trace_user,
  )

  configure_trace_defaults(
      user={"id": "user-123", "name": "Ada"},
      organization={"id": "org-7", "name": "Acme"},
      metadata={"env": "staging"},
      evaluators=["accuracy"],
  )
  add_default_evaluators("toxicity")
  ```

- Enrich the active span at runtime using helpers or span handles:

  ```python
  with trace_span("custom.section") as span:
      span.add_evaluator("style-guide")
      span.set_experiment("exp-1", feature_slug="agent")

  set_trace_user("user-999", name="Grace")
  set_trace_organization("org-15", name="Basalt Labs")
  attach_trace_experiment("exp-2", name="A/B test", feature_slug="beta-agent")
  ```

**Supported environment variables:**

| Variable | Description |
| --- | --- |
| `BASALT_API_KEY` | API key for authentication (can also be passed to `Basalt()` constructor). |
| `BASALT_TELEMETRY_ENABLED` | Master switch to enable/disable telemetry (default: `true`). |
| `BASALT_SERVICE_NAME` | Overrides the OTEL `service.name`. |
| `BASALT_ENVIRONMENT` | Sets `deployment.environment`. |
| `BASALT_OTEL_EXPORTER_OTLP_ENDPOINT` | Custom OTLP HTTP endpoint for traces. Overrides the default Basalt OTEL collector endpoint. |
| `BASALT_BUILD` | SDK build mode - set to `development` for local OTEL collector testing (default: `production`). |
| `TRACELOOP_TRACE_CONTENT` | Controls whether prompts/completions are logged. **Note:** Set automatically by `TelemetryConfig.llm_trace_content` - you typically don't need to set this manually. |

**Default OTLP Exporter:**

By default, the SDK automatically sends traces to Basalt's OTEL collector:
- **Production**: `https://otel.getbasalt.ai/v1/traces`
- **Development**: `http://localhost:4318/v1/traces` (when `BASALT_BUILD=development`)

You can override this by:
1. Providing a custom `exporter` in `TelemetryConfig`
2. Setting the `BASALT_OTEL_EXPORTER_OTLP_ENDPOINT` environment variable
3. Disabling telemetry with `enable_telemetry=False`

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

#### Span utilities (no OpenTelemetry import required)

For common span operations, you can use lightweight helpers from `basalt.observability` instead of importing `opentelemetry.trace`:

```python
from basalt.observability import (
    trace_span,
    get_current_span,
    get_current_span_handle,
    set_span_attribute,
    set_span_attributes,
    add_span_event,
    record_span_exception,
    set_span_status_ok,
    set_span_status_error,
)

# Create a span and enrich it using helpers
with trace_span("my.workflow"):
    set_span_attribute("workflow.step", 1)
    set_span_attributes({"feature": "demo", "stage": "start"})
    add_span_event("initialized", {"user": "alice"})

    try:
        # your logic here
        set_span_status_ok("done")
    except Exception as exc:
        record_span_exception(exc)
        set_span_status_error(str(exc))

# You can also access the active span directly
span = get_current_span()          # returns an OTEL Span or None
handle = get_current_span_handle() # returns a SpanHandle or None
```

## License

MIT
