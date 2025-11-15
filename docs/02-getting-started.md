# Getting Started

This guide will walk you through installing the Basalt SDK, configuring your environment, and making your first API calls.

## Installation

### Basic Installation

```bash
pip install basalt-sdk
```

### With LLM Provider Instrumentation

The Basalt SDK supports automatic instrumentation for various LLM providers, vector databases, and AI frameworks. You can install only the instrumentations you need using optional dependencies:

```bash
# Install with specific LLM provider instrumentation
pip install basalt-sdk[openai]
pip install basalt-sdk[anthropic]

# Install with multiple providers
pip install basalt-sdk[openai,anthropic]

# Install all LLM providers
pip install basalt-sdk[llm-all]

# Install all instrumentations (LLM + vector databases + frameworks)
pip install basalt-sdk[all]
```

#### Available Provider Groups

| Group                        | Description                          | Providers Included                                                                                                             |
|------------------------------|--------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| **Individual LLM Providers** | Single provider instrumentation      | `openai`, `anthropic`, `google-generativeai`, `cohere`, `bedrock`, `vertex-ai`, `ollama`, `mistralai`, `together`, `replicate` |
| `llm-all`                    | All LLM providers                    | All 10 LLM providers above                                                                                                     |
| **Vector Databases**         | Individual vector DB instrumentation | `chromadb`, `pinecone`, `qdrant`                                                                                               |
| `vector-all`                 | All vector databases                 | All 3 vector databases above                                                                                                   |
| **Frameworks**               | Individual framework instrumentation | `langchain`, `llamaindex`, `haystack`                                                                                          |
| `framework-all`              | All frameworks                       | All 3 frameworks above                                                                                                         |
| `all`                        | Everything                           | All providers, vector databases, and frameworks                                                                                |

**Note:** You still need to install the actual provider SDKs separately (e.g., `pip install openai` for OpenAI). The basalt-sdk extras install the **instrumentation packages** that enable automatic tracing.

For more details on auto-instrumentation, see [Auto Instrumentation](09-auto-instrumentation.md).

### Verify Installation

```python
import basalt
print(f"Basalt SDK version: {basalt.__version__}")
```

## API Key Setup

### Obtaining an API Key

1. Sign up at [https://getbasalt.ai](https://getbasalt.ai)
2. Navigate to Settings → API Keys
3. Create a new API key
4. Copy the key (it will only be shown once)

### Setting Up Your API Key

#### Option 1: Environment Variable (Recommended)

```bash
export BASALT_API_KEY="your-api-key-here"
```

Add this to your `.bashrc`, `.zshrc`, or `.env` file for persistence.

#### Option 2: Direct in Code

```python
from basalt import Basalt

basalt = Basalt(api_key="your-api-key-here")
```

**Warning**: Never commit API keys to version control. Use environment variables or secret management systems in production.

## Basic Configuration

### Minimal Setup

The simplest way to get started:

```python
from basalt import Basalt

# Initialize with just an API key
# This enables prompt and dataset APIs, but not telemetry
basalt = Basalt(api_key="your-api-key")

# Use the SDK
prompt = basalt.prompts.get_sync(slug='my-prompt', tag='latest')

# Always shutdown when done to release resources
basalt.shutdown()
```

### Enable Telemetry

To enable observability features:

```python
from basalt import Basalt

# Telemetry is enabled by default if TelemetryConfig is provided
basalt = Basalt(
    api_key="your-api-key",
    enable_telemetry=True  # Explicitly enable telemetry
)

# Now you can use observability features
from basalt.observability import observe_span

@observe_span(name="my.operation")
def my_function():
    return "Hello, World!"

my_function()

basalt.shutdown()
```

### Using Context Manager (Recommended)

Automatically handle shutdown:

```python
from basalt import Basalt

# Context manager ensures proper cleanup
with Basalt(api_key="your-api-key", enable_telemetry=True) as basalt:
    prompt = basalt.prompts.get_sync(slug='my-prompt', tag='latest')
    print(prompt.text)

# Automatically calls basalt.shutdown()
```

## Configuration Options

### Basic Client Configuration

```python
from basalt import Basalt

basalt = Basalt(
    api_key="your-api-key",

    # Telemetry control
    enable_telemetry=True,  # Enable observability features

    # Global trace context (applied to all spans)
    trace_metadata={"environment": "production", "region": "us-west"},
)
```

### Telemetry Configuration

For advanced telemetry setup:

```python
from basalt import Basalt, TelemetryConfig

telemetry_config = TelemetryConfig(
    # Service identification
    service_name="my-ai-application",
    service_version="1.0.0",
    environment="production",

    # LLM instrumentation
    enable_llm_instrumentation=True,  # Auto-instrument LLM providers
    llm_trace_content=True,  # Include prompts and completions in traces

    # Selective provider instrumentation
    llm_enabled_providers=["openai", "anthropic"],  # Only these providers
    # OR
    llm_disabled_providers=["langchain"],  # All except these

    # Additional resource attributes
    extra_resource_attributes={
        "team": "ai-research",
        "cost_center": "engineering"
    }
)

basalt = Basalt(
    api_key="your-api-key",
    telemetry_config=telemetry_config
)
```

### Custom OTLP Exporter

To send telemetry to your own collector:

```python
from basalt import Basalt, TelemetryConfig
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Create custom exporter
custom_exporter = OTLPSpanExporter(
    endpoint="https://your-collector.example.com:4317",
    headers={"authorization": "Bearer your-token"},
    insecure=False  # Use TLS
)

telemetry_config = TelemetryConfig(
    service_name="my-app",
    exporter=custom_exporter,
    enable_llm_instrumentation=True
)

basalt = Basalt(
    api_key="your-api-key",
    telemetry_config=telemetry_config
)
```

## Environment Variables Reference

Basalt respects these environment variables:

| Variable                             | Description                                      | Default            |
|--------------------------------------|--------------------------------------------------|--------------------|
| `BASALT_API_KEY`                     | Your Basalt API key                              | **Required**       |
| `BASALT_BUILD`                       | Environment mode (`development` or `production`) | `production`       |
| `BASALT_TELEMETRY_ENABLED`           | Master switch for telemetry                      | `true`             |
| `BASALT_SERVICE_NAME`                | Override service name for OTEL                   | `basalt-sdk`       |
| `BASALT_ENVIRONMENT`                 | Deployment environment (e.g., `staging`)         | None               |
| `BASALT_OTEL_EXPORTER_OTLP_ENDPOINT` | Custom OTLP endpoint                             | Basalt's collector |

**Note:** `TRACELOOP_TRACE_CONTENT` is automatically set by the SDK based on `TelemetryConfig.llm_trace_content`. You typically don't need to set it manually. Use `TelemetryConfig(llm_trace_content=False)` instead.


## Your First API Call

### Example 1: Get a Prompt

```python
from basalt import Basalt

# Initialize
basalt = Basalt(api_key="your-api-key")

# Get a prompt
prompt = basalt.prompts.get_sync(
    slug='customer-greeting',
    tag='production'
)

# Access prompt properties
print(f"Prompt Text: {prompt.text}")
print(f"Model Provider: {prompt.model.provider}")
print(f"Model Name: {prompt.model.model}")
print(f"Temperature: {prompt.model.parameters.temperature}")

# Cleanup
basalt.shutdown()
```

### Example 2: Get a Prompt with Variables

```python
from basalt import Basalt

basalt = Basalt(api_key="your-api-key")

# Get a prompt with variable substitution
prompt = basalt.prompts.get_sync(
    slug='personalized-email',
    tag='latest',
    variables={
        'customer_name': 'Alice',
        'product_name': 'Premium Plan',
        'discount_amount': '20%'
    }
)

# The prompt.text now has variables substituted
print(prompt.text)

basalt.shutdown()
```

### Example 3: List Available Prompts

```python
from basalt import Basalt

basalt = Basalt(api_key="your-api-key")

# List all prompts
response = basalt.prompts.list_sync()

print(f"Total prompts: {response.total}")

for prompt in response.prompts:
    print(f"- {prompt.slug}: {prompt.description}")

basalt.shutdown()
```

### Example 4: Create Your First Trace

```python
from basalt import Basalt
from basalt.observability import observe_span

# Initialize with telemetry
basalt = Basalt(api_key="your-api-key", enable_telemetry=True)

@observe_span(name="hello.world")
def greet(name: str):
    return f"Hello, {name}!"

# This function call will create a trace
result = greet("Alice")
print(result)

basalt.shutdown()
```

## Complete Quickstart Example

Here's a complete example that demonstrates prompts + telemetry:

```python
from basalt import Basalt, TelemetryConfig
from basalt.observability import observe_span, observe_generation
import openai
import os

# Initialize Basalt
basalt = Basalt(
    api_key=os.getenv("BASALT_API_KEY"),
    telemetry_config=TelemetryConfig(
        service_name="quickstart-demo",
        environment="development",
        enable_llm_instrumentation=True
    )
)

# Initialize OpenAI
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@observe_span(name="demo.workflow")
def run_demo(user_query: str):
    """Complete workflow with prompt retrieval and LLM call"""

    # Step 1: Get prompt from Basalt
    prompt_template = basalt.prompts.get_sync(
        slug='demo-prompt',
        tag='latest',
        variables={'query': user_query}
    )

    # Step 2: Call OpenAI (automatically traced)
    response = openai_client.chat.completions.create(
        model=prompt_template.model.model,
        messages=[{"role": "user", "content": prompt_template.text}],
        temperature=prompt_template.model.parameters.temperature
    )

    return response.choices[0].message.content

# Run the demo
try:
    result = run_demo("What is the capital of France?")
    print(f"Result: {result}")
finally:
    # Always shutdown to flush telemetry
    basalt.shutdown()
```

## Verification

### Check if API Key is Working

```python
from basalt import Basalt
from basalt.types.exceptions import UnauthorizedError

try:
    basalt = Basalt(api_key="your-api-key")
    prompts = basalt.prompts.list_sync()
    print(f"✓ API key is valid. Found {prompts.total} prompts.")
    basalt.shutdown()
except UnauthorizedError:
    print("✗ API key is invalid or expired.")
except Exception as e:
    print(f"✗ Error: {e}")
```

### Check if Telemetry is Working

```python
from basalt import Basalt
from basalt.observability import observe_span, get_current_span

basalt = Basalt(api_key="your-api-key", enable_telemetry=True)

@observe_span(name="telemetry.test")
def test_telemetry():
    span = get_current_span()
    if span and span.is_recording():
        print("✓ Telemetry is working!")
        return True
    else:
        print("✗ Telemetry is not recording.")
        return False

test_telemetry()
basalt.shutdown()
```
---