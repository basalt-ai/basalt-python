# Observability

Basalt provides a powerful and intuitive observability API designed to help you understand your application's behavior, performance, and usage.

## The `observe` API

The core of Basalt's observability is the `observe` class. It unifies tracing, logging, and context management into a single, easy-to-use interface.

### Spans and Kinds

Every operation you track is recorded as a **Span**. Spans can have a `kind` that describes their semantic meaning.

```python
from basalt.observability import observe, start_observe, ObserveKind

# Start a trace (Root Span)
@start_observe(name="process_request")
def process():
    pass

# Nested span
@observe(name="sub_task")
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

### Enriching Spans

You can enrich spans with various types of data using static methods on `observe`. These methods always apply to the *current active span*.

#### Identity (`observe.identify`)

Associate traces with users and organizations to track usage and costs.

```python
observe.identify(user="user_123", organization="org_abc")
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
observe.input({"query": "hello"})
observe.output("world")
```

#### Status (`observe.status`, `observe.fail`)

Mark spans as successful or failed to track operation outcomes.

**When to use:**
- **`observe.status("ok")`** - Mark an operation as successful (optional, spans default to OK)
- **`observe.status("error", "message")`** - Mark an operation as failed with a description
- **`observe.fail(exception)`** - Record an exception and automatically mark the span as failed

**Business value:** Status tracking helps you identify failure patterns, calculate success rates, and prioritize reliability improvements.

```python
# Mark success (usually implicit)
observe.status("ok")

# Mark explicit failure with context
observe.status("error", "Payment gateway timeout")

# Record exception with full details
try:
    process_payment(amount)
except PaymentError as e:
    observe.fail(e)  # Captures exception details and sets error status
    raise
```

**Best practices:**
- Let exceptions bubble up naturally - they'll be captured automatically
- Use `observe.fail()` when you want to record the error but continue execution
- Add descriptive error messages to help with debugging later

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
from basalt.observability import observe

@observe(name="API Handler")
def handle_request(user_id):
    # Root span starts here
    authenticate(user_id)
    process_data()

@observe(name="Authenticate")
def authenticate(user_id):
    # Verify credentials...
    pass

@observe(name="Process Data")
def process_data():
    # Deep in the call stack, we want to tag the root span
    root = observe.root_span()
    if root:
        root.set_attribute("processing_stage", "data_validation")
    
    # Or use identify() which works on the current context
    observe.identify(user="user_123")
```

**When to use `root_span()`:**
- Setting metadata on the top-level operation from nested functions
- Accessing the parent trace context for late-binding operations
- Coordinating behavior across the entire trace hierarchy

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
