# Observability Basics

Basalt provides comprehensive observability for AI applications built on OpenTelemetry, the industry standard for distributed tracing. This guide covers the fundamentals of telemetry configuration and tracing concepts.

## What is OpenTelemetry?

**OpenTelemetry (OTEL)** is an open-source observability framework that provides:
- **Traces**: Track requests as they flow through your system
- **Spans**: Individual units of work within a trace
- **Attributes**: Key-value pairs that describe spans
- **Events**: Timestamped records within spans

### Why OpenTelemetry?

- **Industry Standard**: Vendor-neutral, widely adopted
- **No Lock-in**: Export to any OTLP-compatible backend
- **Rich Ecosystem**: Integrates with hundreds of tools
- **Future-proof**: Active development and community support

## Telemetry Configuration

### Basic Setup

```python
from basalt import Basalt

# Minimal setup - telemetry disabled
basalt = Basalt(api_key="your-api-key")

# Enable telemetry
basalt = Basalt(
    api_key="your-api-key",
    enable_telemetry=True  # Enables observability features
)
```

### Advanced Configuration

```python
from basalt import Basalt, TelemetryConfig

telemetry_config = TelemetryConfig(
    # Service identification
    service_name="my-ai-app",           # Name shown in traces
    service_version="1.2.0",            # App version
    environment="production",            # Environment label

    # LLM instrumentation
    enable_llm_instrumentation=True,    # Auto-trace LLM calls
    llm_trace_content=True,             # Include prompts/completions

    # Resource attributes (appear on all spans)
    extra_resource_attributes={
        "team": "ai-platform",
        "cost_center": "engineering",
        "deployment.region": "us-west-2"
    }
)

basalt = Basalt(
    api_key="your-api-key",
    telemetry_config=telemetry_config
)
```

## TelemetryConfig Options

### Service Identification

```python
TelemetryConfig(
    service_name="customer-support-bot",
    service_version="2.1.0",
    environment="staging"
)
```

These appear as **resource attributes** on all spans:
- `service.name`: Identifies your application
- `service.version`: Tracks deployments
- `deployment.environment`: Separates prod/staging/dev

### LLM Instrumentation Control

```python
# Enable auto-instrumentation for all supported providers
TelemetryConfig(
    enable_llm_instrumentation=True
)

# Selective instrumentation - only these providers
TelemetryConfig(
    enable_llm_instrumentation=True,
    llm_enabled_providers=["openai", "anthropic"]
)

# Exclude specific providers
TelemetryConfig(
    enable_llm_instrumentation=True,
    llm_disabled_providers=["langchain"]  # Skip langchain
)
```

**Supported Providers** (17 total):

*LLM Providers (10):*
- `openai` - OpenAI API
- `anthropic` - Anthropic Claude
- `google_generativeai` - Google Generative AI (Gemini)
- `cohere` - Cohere API
- `bedrock` - AWS Bedrock
- `vertexai` / `vertex-ai` - Google Vertex AI
- `ollama` - Ollama
- `mistralai` - Mistral AI
- `together` - Together AI
- `replicate` - Replicate

**Note:** The NEW Google GenAI SDK (`google_genai`) instrumentation is in the code but the instrumentation package is not yet available on PyPI. Use `google_generativeai` (LEGACY SDK) for now.

*Vector Databases (3):*
- `chromadb` - ChromaDB
- `pinecone` - Pinecone
- `qdrant` - Qdrant

*Frameworks (3):*
- `langchain` - LangChain
- `llamaindex` - LlamaIndex
- `haystack` - Haystack

**Installation:** To use auto-instrumentation, install the instrumentation packages using optional dependencies:
```bash
pip install basalt-sdk[openai,anthropic]  # Specific providers
pip install basalt-sdk[llm-all]            # All LLM providers
pip install basalt-sdk[all]                # Everything
```

See [Getting Started](02-getting-started.md#with-llm-provider-instrumentation) for installation details and [Auto Instrumentation](09-auto-instrumentation.md) for configuration options.

### Content Tracing

```python
# Include prompts and completions in traces (default: True)
TelemetryConfig(
    llm_trace_content=True
)

# Exclude sensitive content for privacy/cost reasons
TelemetryConfig(
    llm_trace_content=False
)
```

**When to disable**:
- Handling sensitive/PII data
- Reducing trace payload size
- Complying with data retention policies

### Custom Resource Attributes

```python
TelemetryConfig(
    extra_resource_attributes={
        "team": "ai-research",
        "cost_center": "product",
        "git.commit": "abc123",
        "k8s.pod.name": "ai-worker-3"
    }
)
```

These attributes appear on **every span** and help with:
- Filtering traces by team/service
- Cost attribution
- Deployment tracking

## Custom OTLP Exporters

### Using Your Own Collector

```python
from basalt import Basalt, TelemetryConfig
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Create custom exporter
exporter = OTLPSpanExporter(
    endpoint="https://your-collector.example.com:4317",
    headers={
        "authorization": "Bearer your-token",
    },
    insecure=False  # Use TLS
)

# Use custom exporter
telemetry_config = TelemetryConfig(
    service_name="my-app",
    exporter=exporter,
    enable_llm_instrumentation=True
)

basalt = Basalt(
    api_key="your-api-key",
    telemetry_config=telemetry_config
)
```


## Environment Variables

Basalt respects these environment variables for telemetry:

| Variable                             | Description                                      | Default                 |
|--------------------------------------|--------------------------------------------------|-------------------------|
| `BASALT_API_KEY`                     | API key for Basalt authentication                | None (required)         |
| `BASALT_TELEMETRY_ENABLED`           | Master telemetry switch                          | `true`                  |
| `BASALT_SERVICE_NAME`                | Override service name                            | `basalt-sdk`            |
| `BASALT_ENVIRONMENT`                 | Deployment environment (prod/staging/dev)        | None                    |
| `BASALT_OTEL_EXPORTER_OTLP_ENDPOINT` | Custom OTLP endpoint for traces                  | Auto (Basalt collector) |
| `BASALT_BUILD`                       | SDK build mode (`development` for local testing) | `production`            |


### Example .env File

```
# Basalt configuration
BASALT_API_KEY=bsk_1234567890
BASALT_TELEMETRY_ENABLED=true
BASALT_SERVICE_NAME=my-app
BASALT_ENVIRONMENT=production

# Custom endpoint (optional)
BASALT_OTEL_EXPORTER_OTLP_ENDPOINT=https://my-collector.com:4317
```

### Using Environment Variables

```python
import os
from basalt import Basalt, TelemetryConfig

# Environment variables are automatically picked up
basalt = Basalt(
    api_key=os.getenv("BASALT_API_KEY"),
    telemetry_config=TelemetryConfig(
        service_name=os.getenv("BASALT_SERVICE_NAME", "default-service"),
        environment=os.getenv("BASALT_ENVIRONMENT", "development")
    )
)
```

## Default OTLP Endpoints

Basalt automatically configures the OTLP endpoint based on the environment:

### Production (default)

```
Endpoint: https://otel.getbasalt.ai/v1/traces
Protocol: HTTP/protobuf
Authentication: Via API key
```

### Development

Set `BASALT_BUILD=development`:

```
Endpoint: http://localhost:4318/v1/traces
Protocol: HTTP/protobuf
No authentication required
```

This allows local development with your own collector.

## Understanding Traces and Spans

### Trace Hierarchy

```
Trace (unique ID for entire request)
└── Root Span (e.g., "handle_user_request")
    ├── Child Span (e.g., "retrieve_documents")
    │   └── Grandchild Span (e.g., "vector_db.search")
    └── Child Span (e.g., "generate_response")
        └── Grandchild Span (e.g., "openai.chat.completion")
```

### Span Anatomy

Each span contains:

```python
{
    "name": "llm.generate",
    "trace_id": "abc123...",
    "span_id": "def456...",
    "parent_span_id": "ghi789...",
    "start_time": "2025-01-15T10:30:00Z",
    "end_time": "2025-01-15T10:30:02Z",
    "duration_ms": 2000,

    "attributes": {
        "llm.model": "gpt-4",
        "llm.provider": "openai",
        "llm.usage.prompt_tokens": 50,
        "llm.usage.completion_tokens": 100,
        "user.id": "user-123",
        "organization.id": "org-456"
    },

    "events": [
        {"name": "cache_miss", "timestamp": "..."},
        {"name": "api_call_started", "timestamp": "..."}
    ],

    "status": "OK"  # or "ERROR"
}
```

## Basic Tracing Example

### Creating Your First Span

```python
from basalt import Basalt
from basalt.observability import observe_span

# Initialize with telemetry
basalt = Basalt(api_key="your-api-key", enable_telemetry=True)

@observe_span(name="my_app.process_request")
def process_request(user_id: str, data: dict):
    """Process a user request with tracing"""

    # Your business logic here
    result = {"status": "success", "user_id": user_id}

    return result

# Call the function - automatically creates a trace
result = process_request(user_id="user-123", data={"query": "hello"})

# Shutdown to flush traces
basalt.shutdown()
```

### Nested Spans Example

```python
from basalt import Basalt
from basalt.observability import observe_span

basalt = Basalt(api_key="your-api-key", enable_telemetry=True)

@observe_span(name="app.workflow")
def workflow(input_data: str):
    """Main workflow with nested operations"""

    # This creates child spans automatically
    step1_result = preprocessing(input_data)
    step2_result = processing(step1_result)
    step3_result = postprocessing(step2_result)

    return step3_result

@observe_span(name="app.preprocessing")
def preprocessing(data: str):
    return data.strip().lower()

@observe_span(name="app.processing")
def processing(data: str):
    return data.upper()

@observe_span(name="app.postprocessing")
def postprocessing(data: str):
    return f"Result: {data}"

# Creates a trace with 4 spans (1 parent, 3 children)
result = workflow("  Hello World  ")

basalt.shutdown()
```

## Viewing Your Traces

After sending traces to Basalt:

1. **Log in** to the Basalt dashboard at https://app.getbasalt.ai
2. Navigate to **Traces** section
3. Filter by:
   - Service name
   - Environment
   - User/Organization
   - Time range
4. Click on a trace to see:
   - Span hierarchy (waterfall view)
   - Attributes and events
   - LLM usage (prompts, completions, tokens)
   - Evaluator results

## Best Practices

### 1. Use Descriptive Service Names

```python
# Good
TelemetryConfig(service_name="customer-support-chatbot")

# Avoid
TelemetryConfig(service_name="app")
```

### 2. Set Appropriate Environments

```python
# Development
TelemetryConfig(environment="development")

# Staging
TelemetryConfig(environment="staging")

# Production
TelemetryConfig(environment="production")
```

### 3. Always Call shutdown()

```python
# Bad: Spans may not flush
basalt = Basalt(api_key="key", enable_telemetry=True)
do_work()
# Missing: basalt.shutdown()

# Good: Use context manager
with Basalt(api_key="key", enable_telemetry=True) as basalt:
    do_work()
# Automatic shutdown

# Or explicit shutdown
basalt = Basalt(api_key="key", enable_telemetry=True)
try:
    do_work()
finally:
    basalt.shutdown()
```

### 4. Add Meaningful Resource Attributes

```python
TelemetryConfig(
    extra_resource_attributes={
        "team": "platform",
        "service.tier": "premium",
        "deployment.id": deployment_id
    }
)
```


## Architecture: How Environment Variables Work

Understanding how the SDK handles environment variables can help you configure it effectively:

### Two Types of Environment Variables

**1. Variables READ by Basalt SDK:**
- `BASALT_API_KEY` - Authentication
- `BASALT_TELEMETRY_ENABLED` - Enable/disable telemetry
- `BASALT_SERVICE_NAME` - Service identification
- `BASALT_ENVIRONMENT` - Deployment environment
- `BASALT_OTEL_EXPORTER_OTLP_ENDPOINT` - Custom OTLP endpoint
- `BASALT_BUILD` - Development vs production mode

These are read from the environment and can override `TelemetryConfig` settings.