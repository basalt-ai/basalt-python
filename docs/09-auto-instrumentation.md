# Auto Instrumentation

The Basalt SDK provides automatic instrumentation for various LLM providers, vector databases, and AI frameworks through OpenTelemetry. This allows you to capture detailed traces of your AI application's behavior without manually adding instrumentation code.

## Overview

When you enable LLM instrumentation in the Basalt SDK, it automatically instruments supported providers that you have installed. The SDK uses a flexible approach where it only instruments the providers whose packages are actually installed in your environment.

## Installation

The SDK supports optional dependencies for different providers. You can install only the instrumentations you need:

### LLM Provider Instrumentations

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

# All LLM providers (all 10 available providers)
pip install basalt-sdk[llm-all]
```

**Note:** The NEW Google GenAI SDK instrumentation (`google-genai`) is not yet available on PyPI. Use `google-generativeai` for now.

### Vector Database Instrumentations

```bash
# Individual vector databases
pip install basalt-sdk[chromadb]
pip install basalt-sdk[pinecone]
pip install basalt-sdk[qdrant]

# All vector databases
pip install basalt-sdk[vector-all]
```

### Framework Instrumentations

```bash
# Individual frameworks
pip install basalt-sdk[langchain]
pip install basalt-sdk[llamaindex]
pip install basalt-sdk[haystack]

# All frameworks
pip install basalt-sdk[framework-all]
```

### Install Everything

```bash
# Install all available instrumentations
pip install basalt-sdk[all]
```

## Basic Usage

Auto-instrumentation is enabled by default. You can control it using the `enable_instrumentation` setting:

```python
from basalt import Basalt, TelemetryConfig

# Enable auto-instrumentation for all installed providers (default behavior)
telemetry = TelemetryConfig(
    service_name="my-app",
    enable_instrumentation=True,
)

basalt = Basalt(api_key="your-api-key", telemetry_config=telemetry)

# Or use client-level parameters for simplicity
basalt = Basalt(
    api_key="your-api-key",
    enabled_instruments=["openai", "anthropic"]
)

# Now your LLM calls will be automatically traced
import openai
client = openai.OpenAI()

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
# This call is automatically traced with input/output, token counts, etc.
```

## Supported Providers

### LLM Providers

The SDK automatically instruments the following LLM providers when their packages are installed (10 available + 1 in code):

- **openai** - OpenAI API (via `opentelemetry-instrumentation-openai`)
- **anthropic** - Anthropic API (via `opentelemetry-instrumentation-anthropic`)
- **google_generativeai** - Google Generative AI SDK / Gemini (via `opentelemetry-instrumentation-google-generativeai`)
  - Use for: `import google.generativeai`
- **google_genai** - Google GenAI SDK - NEW (instrumentation not yet available on PyPI)
  - Code ready but package not published yet
  - Will use: `from google import genai`
- **cohere** - Cohere API (via `opentelemetry-instrumentation-cohere`)
- **bedrock** - AWS Bedrock (via `opentelemetry-instrumentation-bedrock`)
- **vertexai** / **vertex-ai** - Google Vertex AI (via `opentelemetry-instrumentation-vertexai`)
  - Both names work as aliases
- **ollama** - Ollama (via `opentelemetry-instrumentation-ollama`)
- **mistralai** - Mistral AI (via `opentelemetry-instrumentation-mistralai`)
- **together** - Together AI (via `opentelemetry-instrumentation-together`)
- **replicate** - Replicate (via `opentelemetry-instrumentation-replicate`)

### Vector Databases

The SDK automatically instruments vector database operations (3 total):

- **chromadb** - ChromaDB (via `opentelemetry-instrumentation-chromadb`)
- **pinecone** - Pinecone (via `opentelemetry-instrumentation-pinecone`)
- **qdrant** - Qdrant (via `opentelemetry-instrumentation-qdrant`)

### AI Frameworks

The SDK automatically instruments popular AI frameworks (3 total):

- **langchain** - LangChain (via `opentelemetry-instrumentation-langchain`)
- **llamaindex** - LlamaIndex (via `opentelemetry-instrumentation-llamaindex`)
- **haystack** - Haystack (via `opentelemetry-instrumentation-haystack`)

**Total: 17 providers supported** (10 LLM + 3 Vector DB + 3 Frameworks + 1 in code not yet available)

## Selective Instrumentation

You can control which instruments are enabled using client-level parameters or TelemetryConfig options:

### Using Client-Level Parameters (Recommended)

```python
from basalt import Basalt

# Only instrument OpenAI and Anthropic
basalt = Basalt(
    api_key="your-api-key",
    enabled_instruments=["openai", "anthropic"]
)

# Or disable specific providers
basalt = Basalt(
    api_key="your-api-key",
    disabled_instruments=["langchain", "llamaindex"]
)
```

### Using TelemetryConfig

```python
from basalt import Basalt, TelemetryConfig

# Only instrument OpenAI and Anthropic
telemetry = TelemetryConfig(
    enable_instrumentation=True,
    enabled_providers=["openai", "anthropic"],
)

basalt = Basalt(api_key="your-api-key", telemetry_config=telemetry)

# Instrument all available providers except LangChain
telemetry = TelemetryConfig(
    enable_instrumentation=True,
    disabled_providers=["langchain"],
)

basalt = Basalt(api_key="your-api-key", telemetry_config=telemetry)
```

## How It Works

The Basalt SDK uses a smart approach to auto-instrumentation:

1. **Dynamic Loading**: When you enable LLM instrumentation, the SDK attempts to load instrumentation packages for all supported providers.

2. **Graceful Degradation**: If a provider's instrumentation package is not installed, the SDK silently skips it and logs a debug message. Your application continues to work normally.

3. **No Double Instrumentation**: The SDK checks if a provider is already instrumented before attempting to instrument it again, avoiding conflicts.

4. **Provider Detection**: The SDK only instruments providers whose packages are actually available in your environment.


## Complete Example

Here's a complete example showing auto-instrumentation with multiple providers:

```python
from basalt import Basalt, TelemetryConfig
import openai
import anthropic

# Configure telemetry with auto-instrumentation
telemetry = TelemetryConfig(
    service_name="multi-llm-app",
    environment="production",
    enable_instrumentation=True,
    trace_content=True,
    enabled_providers=["openai", "anthropic"],  # Only these two
)

# Initialize Basalt
basalt = Basalt(api_key="your-basalt-api-key", telemetry_config=telemetry)

# Use OpenAI - automatically traced
openai_client = openai.OpenAI(api_key="your-openai-key")
openai_response = openai_client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello from OpenAI!"}]
)
print(openai_response.choices[0].message.content)

# Use Anthropic - automatically traced
anthropic_client = anthropic.Anthropic(api_key="your-anthropic-key")
anthropic_response = anthropic_client.messages.create(
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello from Anthropic!"}]
)
print(anthropic_response.content[0].text)

# Shutdown to flush traces
basalt.shutdown()
```