# Introduction & Overview

## What is Basalt?

Basalt is a comprehensive platform and Python SDK for building production-ready AI applications. It combines three core capabilities:

1. **Prompt Management** - Version control, retrieval, and A/B testing for AI prompts
2. **Dataset Management** - Test datasets for evaluating AI model outputs
3. **Complete Observability** - OpenTelemetry-based tracing for AI applications with support for 15+ LLM providers

Basalt helps teams move from experimentation to production by providing the infrastructure needed to monitor, evaluate, and continuously improve AI applications.

## Key Features

### Prompt Management
- **Version Control**: Track changes to prompts over time
- **Tag-based Deployment**: Publish prompts to environments (development, staging, production)
- **Variable Substitution**: Dynamic prompts using Jinja2 templates
- **Caching**: Intelligent caching with fallback for high availability

### Dataset Management
- **Test Data Organization**: Structure test cases with inputs and expected outputs
- **Evaluation Support**: Store ideal outputs for automated quality checks
- **Metadata**: Attach context to each test case

### Observability & Tracing
- **OpenTelemetry Standard**: Industry-standard distributed tracing
- **Automatic Instrumentation**: Zero-code monitoring for OpenAI, Anthropic, Gemini, Cohere, and more
- **Manual Tracing**: Fine-grained control with decorators and context managers
- **Semantic Conventions**: Specialized span types for LLM, retrieval, tool, and function calls
- **Evaluator Integration**: Attach quality evaluators to spans with sample rate control

### Experiment Tracking
- **A/B Testing**: Compare model variants and prompts
- **Feature Flags**: Track which features are active in traces
- **User & Organization Attribution**: Associate traces with users and organizations

## Use Cases

### Prompt Engineering & Version Control
```python
# Retrieve the latest version of a prompt
prompt = basalt.prompts.get_sync(
    slug='customer-support-assistant',
    tag='production',
    variables={'product_name': 'Premium Plan'}
)
```

### AI Application Monitoring
```python
# Automatically trace all OpenAI calls
from basalt import Basalt, TelemetryConfig

basalt = Basalt(
    api_key="your-api-key",
    telemetry_config=TelemetryConfig(
        service_name="customer-support",
        enable_instrumentation=True
    )
)

# All OpenAI calls are now automatically traced
response = openai_client.chat.completions.create(...)
```

### RAG Pipeline Observability
```python
from basalt.observability import observe_retrieval, observe_generation

@observe_retrieval(name="knowledge_base.search")
def search_docs(query: str):
    return vector_db.search(query, top_k=5)

@observe_generation(name="llm.answer")
def generate_answer(context: str, question: str):
    return llm.generate(f"Context: {context}\n\nQuestion: {question}")

# Complete observability for your RAG pipeline
docs = search_docs("How do I reset my password?")
answer = generate_answer(docs, "How do I reset my password?")
```

### Quality Monitoring with Evaluators
```python
from basalt.observability import evaluator, observe_generation

@evaluator(slugs=["accuracy", "toxicity"], sample_rate=0.5)
@observe_generation(name="llm.generate_response")
def generate_response(user_query: str):
    return llm.generate(user_query)

# 50% of traces will be evaluated for accuracy and toxicity
```

## Quickstart

### Installation

```bash
pip install basalt-sdk
```

### Basic Setup

```python
from basalt import Basalt

# Initialize the client with your API key
basalt = Basalt(api_key="your-api-key")

# Get a prompt
prompt = basalt.prompts.get_sync(
    slug='my-prompt',
    tag='latest'
)

print(f"Prompt text: {prompt.text}")
print(f"Model: {prompt.model.provider}/{prompt.model.model}")

# Clean up when done
basalt.shutdown()
```

### Quickstart with Telemetry

```python
from basalt import Basalt, TelemetryConfig
from basalt.observability import observe_generation

# Initialize with telemetry enabled
basalt = Basalt(
    api_key="your-api-key",
    telemetry_config=TelemetryConfig(
        service_name="my-ai-app",
        environment="production",
        enable_instrumentation=True  # Auto-trace LLM calls
    )
)

# Use manual tracing for custom logic
@observe_generation(name="customer_support.generate_response")
def handle_customer_query(query: str):
    # Your LLM call here - automatically instrumented
    response = llm.generate(query)
    return response

# Call your function - fully traced
response = handle_customer_query("How do I upgrade my plan?")

# Shutdown to flush telemetry
basalt.shutdown()
```

### Complete Example: Prompt + Telemetry

```python
from basalt import Basalt, TelemetryConfig
from basalt.observability import observe_span, observe_generation
import openai

# Initialize Basalt with full observability
basalt = Basalt(
    api_key="your-basalt-api-key",
    telemetry_config=TelemetryConfig(
        service_name="email-assistant",
        environment="production",
        enable_instrumentation=True
    )
)

# Initialize OpenAI (will be auto-instrumented)
openai_client = openai.OpenAI(api_key="your-openai-key")

@observe_span(name="email_assistant.process_request")
def process_email_request(email_type: str, recipient_name: str):
    # Get the prompt from Basalt
    prompt = basalt.prompts.get_sync(
        slug='email-generator',
        tag='production',
        variables={
            'email_type': email_type,
            'recipient_name': recipient_name
        }
    )

    # Generate response using OpenAI (automatically traced)
    response = openai_client.chat.completions.create(
        model=prompt.model.model,
        messages=[{"role": "user", "content": prompt.text}]
    )

    return response.choices[0].message.content

# Use the function
email_content = process_email_request(
    email_type="welcome",
    recipient_name="Alice"
)

print(email_content)

# Shutdown
basalt.shutdown()
```

This example demonstrates:
- Prompt retrieval with variable substitution
- Automatic OpenAI instrumentation
- Manual span creation for custom logic
- Complete end-to-end observability

## Architecture Overview

### Client Architecture
```
Basalt Client
├── Prompts Client (API interactions)
├── Datasets Client (API interactions)
├── Experiments Client (API interactions)
└── Observability (OpenTelemetry integration)
    ├── Instrumentation Manager (auto-instrumentation)
    ├── Context Processors (trace context)
    ├── Decorators (@observe_generation, @observe_span, @evaluator, etc.)
    ├── Context Managers (with trace_span(), etc.)
    └── Span Utilities (lightweight helpers)
```

## System Requirements

- **Python**: 3.10 or higher
- **Dependencies**:
  - `httpx` for HTTP requests
  - `opentelemetry-api` and `opentelemetry-sdk` for telemetry
  - `openllmetry` for LLM auto-instrumentation
  - `jinja2` for prompt template rendering
