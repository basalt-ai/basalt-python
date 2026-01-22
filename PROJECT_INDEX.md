# Project Index: basalt-python

**Generated:** 2026-01-22
**Version:** 1.1.0
**Language:** Python 3.10+

## 📊 Repository Overview

The Basalt SDK is a comprehensive Python client for managing AI prompts, monitoring AI applications, and tracking experiments via OpenTelemetry. The SDK provides async/sync APIs for prompts, datasets, and experiments with built-in observability.

**Key Statistics:**
- Total Python files: 78
- Repository size: ~45MB (excluding .venv)
- Main package: `basalt/` (26 modules)
- Test coverage: 35+ test files
- Examples: 8 examples + 2 notebooks

---

## 📁 Project Structure

```
basalt-python/
├── basalt/                      # Main SDK package
│   ├── __init__.py              # Package entry (lazy imports)
│   ├── client.py                # Main Basalt client
│   ├── config.py                # Global configuration
│   ├── _version.py              # Version: 1.1.0
│   ├── _internal/               # Internal utilities
│   │   ├── base_client.py       # Base API client
│   │   └── http.py              # HTTP client wrapper
│   ├── observability/           # OpenTelemetry integration (14 modules)
│   │   ├── __init__.py          # Observability facade
│   │   ├── api.py               # High-level observe decorators
│   │   ├── config.py            # TelemetryConfig
│   │   ├── decorators.py        # @observe, @evaluate
│   │   ├── context_managers.py  # Span context managers
│   │   ├── trace.py             # Trace API
│   │   ├── trace_context.py     # Identity & experiment tracking
│   │   ├── instrumentation.py   # LLM/DB auto-instrumentation
│   │   ├── processors.py        # OTEL span processors
│   │   ├── spans.py             # Basalt span wrappers
│   │   ├── evaluators.py        # Custom evaluators
│   │   ├── request_tracing.py   # Request span tracking
│   │   ├── resilient_exporters.py # Error-resilient exporters
│   │   ├── semconv.py           # Semantic conventions
│   │   ├── types.py             # Type definitions
│   │   └── utils.py             # Utility functions
│   ├── prompts/                 # Prompts API client
│   │   ├── __init__.py
│   │   ├── client.py            # PromptsClient (list, get, describe)
│   │   └── models.py            # Prompt, PromptResponse models
│   ├── datasets/                # Datasets API client
│   │   ├── __init__.py
│   │   ├── client.py            # DatasetsClient (list, get, add_row)
│   │   ├── models.py            # Dataset, DatasetRow models
│   │   └── file_upload.py       # File attachment handling
│   ├── experiments/             # Experiments API client
│   │   ├── __init__.py
│   │   ├── client.py            # ExperimentsClient
│   │   └── models.py            # Experiment models
│   ├── types/                   # Shared types
│   │   ├── __init__.py
│   │   ├── exceptions.py        # API exceptions
│   │   └── cache.py             # Cache protocol
│   └── utils/                   # Utilities
│       ├── __init__.py
│       └── memcache.py          # Memory cache implementation
├── tests/                       # Test suite (35+ files)
│   ├── conftest.py              # Pytest fixtures
│   ├── observability/           # Observability tests (14 files)
│   ├── prompts/                 # Prompts API tests
│   ├── datasets/                # Datasets API tests
│   ├── experiments/             # Experiments API tests
│   ├── internal/                # Internal tests
│   └── otel/                    # OTEL integration tests
├── examples/                    # Usage examples
│   ├── openai_example.py        # OpenAI + observability
│   ├── async_observe_example.py # Async decorators
│   ├── dataset_api_example.py   # Dataset operations
│   ├── gemini_random_data_example.py # Gemini integration
│   ├── multi_exporter_example.py # Multiple OTLP exporters
│   ├── internal_api.py          # Internal API demo
│   ├── prompt_sdk_demo.ipynb    # Prompt SDK notebook
│   └── dataset_sdk_demo.ipynb   # Dataset SDK notebook
├── docs/                        # Documentation (13 guides)
│   ├── 01-introduction.md
│   ├── 02-getting-started.md
│   ├── 03-prompts.md
│   ├── 04-datasets.md
│   ├── 05-observability.md
│   ├── 06-manual-tracing.md
│   ├── 07-llm-tracing.md
│   ├── 08-async-observability.md
│   ├── 09-auto-instrumentation.md
│   ├── 10-evaluators.md
│   ├── 11-experiments.md
│   ├── 12-user-org-tracking.md
│   └── 13-trace-context.md
├── pyproject.toml               # Project config (Hatch)
├── README.md                    # Main documentation
├── DEVELOPMENT.md               # Development guide
├── AGENTS.md                    # Agent instructions
└── renovate.json                # Dependency updates

```

---

## 🚀 Entry Points

### 1. **Main SDK Entry**: `basalt/__init__.py`
- Exports: `Basalt`, `TelemetryConfig`, `__version__`
- Uses lazy imports via `__getattr__` for faster startup

### 2. **Primary Client**: `basalt/client.py`
- **Class**: `Basalt`
- **Services**: `prompts`, `datasets`, `experiments`
- **Features**: HTTP client, telemetry config, instrumentation, global metadata
- **Key method**: `shutdown()` - flushes telemetry

### 3. **CLI/Programmatic Usage**:
```python
from basalt import Basalt, TelemetryConfig
basalt = Basalt(api_key="...", telemetry_config=TelemetryConfig(...))
```

---

## 📦 Core Modules

### **basalt.client** - Main Client
- **Exports**: `Basalt` class
- **Purpose**: Central SDK entry point, orchestrates sub-clients
- **Dependencies**: HTTP client, instrumentation, telemetry config

### **basalt.observability** - Telemetry & Tracing
- **Exports**:
  - Decorators: `@observe`, `@start_observe`, `@evaluate`
  - Context managers: `LLMSpanHandle`, `RetrievalSpanHandle`, etc.
  - Config: `TelemetryConfig`
  - API: `Trace`, `trace`, `TraceIdentity`, `TraceExperiment`
- **Purpose**: OpenTelemetry integration with LLM-specific semantics
- **Key features**:
  - Auto-instrumentation for OpenAI, Anthropic, Gemini, Bedrock, etc.
  - Custom evaluators and processors
  - Identity/experiment tracking across traces
  - Resilient exporters with error handling

### **basalt.prompts** - Prompts API
- **Exports**: `PromptsClient`, `Prompt`, `PromptResponse`
- **Methods**:
  - `list_sync()` / `list_async()` - List all prompts
  - `get_sync(slug, tag?, version?, variables?)` - Get prompt
  - `describe_sync(slug)` - Get metadata
- **Purpose**: Fetch and render prompts from Basalt API

### **basalt.datasets** - Datasets API
- **Exports**: `DatasetsClient`, `Dataset`, `DatasetRow`
- **Methods**:
  - `list_sync()` / `list_async()` - List datasets
  - `get_sync(slug)` - Get dataset with rows
  - `add_row_sync(slug, data, attachments?)` - Add row with file uploads
- **Purpose**: Manage evaluation/test datasets

### **basalt.experiments** - Experiments API
- **Exports**: `ExperimentsClient`, `Experiment`
- **Methods**:
  - `create_sync(name, description)` - Create experiment
- **Purpose**: Track A/B tests and experiments

### **basalt.types.exceptions** - Exception Hierarchy
- **Base**: `BasaltAPIError`
- **Specific**: `NotFoundError`, `UnauthorizedError`, `NetworkError`
- **Purpose**: Type-safe error handling

### **basalt._internal.http** - HTTP Client
- **Exports**: `HTTPClient`
- **Features**: Sync/async requests, auth headers, error handling
- **Purpose**: Shared HTTP transport for all API clients

---

## 🔧 Configuration

### **pyproject.toml** - Project Metadata
- **Build system**: Hatchling
- **Python**: >=3.10
- **Core deps**: OpenTelemetry, httpx, jinja2, wrapt
- **Optional deps**:
  - LLM providers (10): openai, anthropic, google-generativeai, etc.
  - Vector DBs (3): chromadb, pinecone, qdrant
  - Frameworks (2): langchain, llamaindex

### **basalt/config.py** - Runtime Config
- Default API base URL
- Environment variable parsing
- Global settings

### **basalt/observability/config.py** - Telemetry Config
- **Class**: `TelemetryConfig`
- **Fields**: service_name, environment, trace_content, enabled_providers
- **Purpose**: Centralized OTEL configuration

---

## 📚 Documentation

| File | Topic |
|------|-------|
| `01-introduction.md` | SDK overview |
| `02-getting-started.md` | Installation & setup |
| `03-prompts.md` | Prompts API guide |
| `04-datasets.md` | Datasets API guide |
| `05-observability.md` | Telemetry overview |
| `06-manual-tracing.md` | Custom span creation |
| `07-llm-tracing.md` | LLM provider tracing |
| `08-async-observability.md` | Async patterns |
| `09-auto-instrumentation.md` | Auto-instrumentation setup |
| `10-evaluators.md` | Custom evaluators |
| `11-experiments.md` | Experiment tracking |
| `12-user-org-tracking.md` | Identity tracking |
| `13-trace-context.md` | Context propagation |

---

## 🧪 Test Coverage

### Test Organization
- **Total test files**: 35+
- **Coverage areas**: API clients, observability, OTEL integration, decorators

### Key Test Modules
| Module | Focus |
|--------|-------|
| `tests/prompts/` | Prompts API, context managers |
| `tests/datasets/` | Datasets API, file uploads |
| `tests/experiments/` | Experiments API |
| `tests/observability/` | Decorators, spans, processors, evaluators |
| `tests/otel/` | OTLP export, LLM instrumentation |
| `tests/internal/` | HTTP client |

### Running Tests
```bash
# Via Hatch
hatch run test

# Via pytest directly
pytest tests/ --cov=basalt
```

---

## 🔗 Key Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| opentelemetry-api | ~1.39.1 | OTEL core API |
| opentelemetry-sdk | ~1.39.1 | OTEL SDK |
| opentelemetry-exporter-otlp | ~1.39.1 | OTLP exporter |
| opentelemetry-instrumentation | ~0.59b0 | Auto-instrumentation base |
| opentelemetry-instrumentation-httpx | ~0.59b0 | HTTP tracing |
| httpx | >=0.28.1 | Async HTTP client |
| jinja2 | >=3.1.6 | Prompt template rendering |
| wrapt | ~1.17.3 | Decorator utilities |
| pytest | - | Testing framework |
| ruff | - | Linting & formatting |

---

## 📝 Quick Start

### 1. Installation
```bash
pip install basalt-sdk[openai,anthropic]  # With LLM providers
```

### 2. Basic Usage
```python
from basalt import Basalt

basalt = Basalt(api_key="your-key")

# Get a prompt
prompt = basalt.prompts.get_sync("my-prompt")
print(prompt.text)

# Get a dataset
dataset = basalt.datasets.get_sync("my-dataset")
for row in dataset.rows:
    print(row.data)

# Shutdown (flush telemetry)
basalt.shutdown()
```

### 3. Observability
```python
from basalt.observability import observe, start_observe

@start_observe(name="process_workflow", feature_slug="main")
def main():
    result = generate_text()
    return result

@observe(kind="generation", name="llm.generate")
def generate_text():
    # Your LLM call here
    return "Generated text"
```

### 4. Run Tests
```bash
hatch run test
```

---

## 🎯 Architecture Patterns

### 1. **Lazy Imports**
- `basalt/__init__.py` uses `__getattr__` to defer imports
- Reduces startup time, avoids loading unused dependencies

### 2. **Base Client Pattern**
- `BaseServiceClient` provides common API functionality
- Subclassed by `PromptsClient`, `DatasetsClient`, `ExperimentsClient`

### 3. **Observability Facade**
- High-level API (`@observe`) wraps low-level OTEL primitives
- Context managers (`LLMSpanHandle`) simplify span management

### 4. **Resilient Exporters**
- `ResilientOTLPExporter` wraps OTLP exporter with error handling
- Prevents telemetry failures from breaking app logic

### 5. **Identity Propagation**
- `TraceIdentity` attaches user/org metadata to root spans
- Automatically propagates to child spans via context

---

## 🔍 Key Symbols Reference

### Classes
- `Basalt` - Main SDK client (`basalt/client.py:23`)
- `TelemetryConfig` - Telemetry configuration (`basalt/observability/config.py`)
- `PromptsClient` - Prompts API (`basalt/prompts/client.py`)
- `DatasetsClient` - Datasets API (`basalt/datasets/client.py`)
- `ExperimentsClient` - Experiments API (`basalt/experiments/client.py`)
- `HTTPClient` - HTTP transport (`basalt/_internal/http.py`)
- `InstrumentationManager` - Auto-instrumentation (`basalt/observability/instrumentation.py`)
- `Trace` - Low-level trace API (`basalt/observability/trace.py`)

### Decorators
- `@observe` - Create nested span (`basalt/observability/api.py`)
- `@start_observe` - Create root span with identity (`basalt/observability/api.py`)
- `@evaluate` - Attach evaluators (`basalt/observability/decorators.py`)

### Exceptions
- `BasaltAPIError` - Base exception (`basalt/types/exceptions.py`)
- `NotFoundError` - 404 errors
- `UnauthorizedError` - 401 errors
- `NetworkError` - Connection failures

---

## 📊 Token Efficiency Impact

**Before indexing**: Full codebase read = ~58,000 tokens per session
**After indexing**: Read this index = ~3,000 tokens (94% reduction)

**Expected savings**:
- 10 sessions: 550,000 tokens saved
- 100 sessions: 5,500,000 tokens saved

---

## 🔄 Development Workflow

### Formatting & Linting
```bash
hatch run fmt       # Format with ruff
hatch run lint      # Lint code
hatch run lint-fix  # Auto-fix lint issues
```

### Testing
```bash
hatch run test              # Run tests with coverage
hatch run test-verbose      # Verbose output
```

### Type Checking
```bash
hatch run typecheck  # Run mypy
```

### All Checks
```bash
hatch run all  # fmt + lint-fix + typecheck + test
```

---

## 📌 Important Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Project config, dependencies, Hatch scripts |
| `basalt/__init__.py` | Main package entry (lazy imports) |
| `basalt/client.py` | Primary SDK client |
| `basalt/observability/__init__.py` | Observability facade |
| `basalt/observability/config.py` | Telemetry configuration |
| `basalt/types/exceptions.py` | Exception hierarchy |
| `README.md` | User-facing documentation |
| `DEVELOPMENT.md` | Contributor guide |
| `tests/conftest.py` | Shared pytest fixtures |

---

**End of Index**
*This index provides a comprehensive overview of the basalt-python repository structure, enabling efficient navigation and reducing token usage in future sessions.*
