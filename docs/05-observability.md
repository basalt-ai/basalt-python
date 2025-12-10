# Observability

Basalt provides a powerful and intuitive observability API designed to help you understand your application's behavior, performance, and usage.

## The `observe` API

The core of Basalt's observability is the `observe` and `start_observe` classes. They unify tracing, logging, and context management into a single, easy-to-use interface.

### Root Spans with `start_observe`

Every trace must start with a **root span** created using `start_observe`. This is the entry point for your observability and supports identity tracking, experiment attachment, and evaluator configuration.

```python
from basalt.observability import start_observe, observe, ObserveKind

# Root span with identity tracking
@start_observe(
    feature_slug="request-processing",
    name="process_request",
    identity={"user": {"id": "user_123", "name": "Alice"}, "organization": {"id": "org_abc"}},
    metadata={"environment": "production", "version": "2.0"}
)
def process():
    # Identity automatically propagates to all child spans
    pass

# Root span with experiment tracking
@start_observe(
    feature_slug="ml-workflow",
    name="ml_workflow",
    experiment={"id": "exp_456", "name": "Model Comparison"},
    identity={"user": "analyst_001"}
)
def run_experiment():
    pass

# Context manager form
with start_observe(
    feature_slug="batch-processing",
    name="batch_job",
    identity={"organization": {"id": "org_xyz", "name": "Acme Corp"}},
    metadata={"job_id": "batch_123"}
):
    # Your code here
    pass
```

**Key features of `start_observe`:**
- **`feature_slug`** (required): A unique identifier for the feature or workflow being traced (e.g., "user-auth", "payment-flow").
- **`name`** (required): A descriptive name for the root span.
- **`identity`**: Dict with `user` and/or `organization` keys for tracking. Can also use simple string format or callables for dynamic resolution.
- **`experiment`**: Dict with `id`, `name`, and `variant` keys for A/B testing and feature tracking.
- **`evaluate_config`**: Configuration for evaluators attached to the root span.
- **`metadata`**: Custom key-value pairs attached to the span.

### Spans and Kinds

Nested operations are tracked using `observe` with different `kind` values that describe their semantic meaning.

```python
from basalt.observability import observe, start_observe, ObserveKind

# Root span (required as entry point)
@start_observe(
    feature_slug="request-handler",
    name="process_request",
    identity={
        "organization": {"id": "123", "name": "ACME"},
        "user": {"id": "456", "name": "John Doe"}
    }
)
def process():
    sub_task()
    return "done"

# Nested span (automatically inherits identity)
@observe(name="sub_task", kind=ObserveKind.SPAN)
def sub_task():
    pass

# Generation (LLM calls)
@observe(name="generate_text", kind=ObserveKind.GENERATION)
def generate():
    pass

# Retrieval (RAG/Vector DB)
@observe(name="search_docs", kind=ObserveKind.RETRIEVAL)
def search():
    pass

# Tool Execution
@observe(name="calculator", kind=ObserveKind.TOOL)
def calculate():
    pass
```

**Important:** Always use `start_observe` for the outermost/root span in your trace. Nested spans use `observe`. Identity, experiment, and other context set on `start_observe` automatically propagates to child spans.

### Enriching Spans

You can enrich spans with various types of data using static methods on `observe`. These methods always apply to the *current active span*.

#### Identity (`observe.set_identity`)

Associate traces with users and organizations to track usage and costs.

```python
observe.set_identity({
    "user": {"id": "user-123", "name": "John Doe"},
    "organization": {"id": "org-456", "name": "ACME Corp"}
})
```

#### Metadata (`observe.metadata`, `observe.update_metadata`)

Attach arbitrary key-value pairs for filtering and analysis.

```python
# Set metadata on current span
observe.metadata({
    "environment": "production",
    "feature_flag": "enabled"
})

# Or using keyword arguments
observe.metadata(environment="production", feature_flag="enabled")

# Merge metadata (updates existing keys without removing others)
observe.update_metadata({"request_id": "abc123"})
```

**Difference between `metadata()` and `update_metadata()`:**
- Both methods add metadata to the current span
- `metadata()` is the standard method for adding key-value pairs
- `update_metadata()` is semantically clearer when you're updating existing metadata
- In practice, both methods work identically - choose based on readability

#### Input & Output (`observe.input`, `observe.output`)

Explicitly capture the input arguments and return values of your operations. When using the `@observe` decorator, this is often handled automatically, but you can override or augment it.

```python
observe.set_input({"query": "hello"})
observe.set_output("world")
```


### Evaluators

You can attach evaluators to spans to automatically grade or analyze the outputs.

```python
from basalt.observability import observe, evaluate

@evaluate("toxicity-check")
@observe(name="chat_response", kind="generation")
def chat(message):
    return llm.generate(message)
```

Or dynamically:

```python
observe.evaluate("helpfulness-score")
```

## Context Propagation

Basalt automatically handles context propagation for nested spans, even across async calls.

```python
@observe(name="parent")
async def parent():
    observe.metadata({"parent_attr": "value"})
    await child()

@observe(name="child")
async def child():
    # This span is automatically a child of "parent"
    pass
```

### Accessing the Root Span

Sometimes you need to set metadata or identity on the root span from deeply nested operations:

```python
from basalt.observability import observe, start_observe


@start_observe(feature_slug="api-handler", name="API Handler")
def handle_request(user_id):
    # Root span starts here with identity tracking
    observe.set_input({"user_id": user_id})
    authenticate(user_id)
    process_data()


@observe(name="Authenticate")
def authenticate(user_id):
    # Verify credentials...
    pass

```


## Global Configuration

You can configure global metadata and other settings when initializing the `Basalt` client.

```python
from basalt import Basalt

client = Basalt(
    api_key="...",
    observability_metadata={
        "service.version": "1.2.3",
        "deployment.region": "us-west-2"
    }
)
```

This metadata will be automatically attached to every span created by the SDK.

## Prompt Context Managers

Prompts can be used as context managers to automatically link LLM calls to the prompt data for enhanced observability.

### How It Works

When you use a prompt as a context manager, it creates an active span that any subsequent LLM calls (via auto-instrumented providers) will automatically link to the prompt's text and model configuration.

```python
from basalt import Basalt
import openai

basalt = Basalt(api_key="your-api-key")
client = openai.OpenAI()

# Prompt span becomes active and scopes LLM calls
with basalt.prompts.get_sync('summary-prompt', tag='production') as prompt:
    # This LLM call automatically links to the prompt data
    response = client.chat.completions.create(
        model=prompt.model.model,
        messages=[{"role": "user", "content": prompt.text}]
    )

basalt.shutdown()
```

### Async Support

Context managers work seamlessly with async/await as well:

```python
import asyncio
from basalt import Basalt
import openai

async def generate_summary():
    basalt = Basalt(api_key="your-api-key")
    client = openai.AsyncOpenAI()
    
    # Async context manager for async LLM calls
    async with await basalt.prompts.get('summary-prompt', tag='production') as prompt:
        response = await client.chat.completions.create(
            model=prompt.model.model,
            messages=[{"role": "user", "content": prompt.text}]
        )
    
    basalt.shutdown()

asyncio.run(generate_summary())
```
