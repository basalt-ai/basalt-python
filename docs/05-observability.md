# Observability

Basalt provides a powerful and intuitive observability API designed to help you understand your application's behavior, performance, and usage.

## The `observe` API

The core of Basalt's observability is the `observe` class. It unifies tracing, logging, and context management into a single, easy-to-use interface.

### Spans and Kinds

Every operation you track is recorded as a **Span**. Spans can have a `kind` that describes their semantic meaning.

```python
from basalt.observability import observe, ObserveKind

# Generic span (default)
@observe(name="process_request")
def process():
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

#### Metadata (`observe.metadata`)

Attach arbitrary key-value pairs for filtering and analysis.

```python
observe.metadata({
    "environment": "production",
    "feature_flag": "enabled"
})
```

#### Input & Output (`observe.input`, `observe.output`)

Explicitly capture the input arguments and return values of your operations. When using the `@observe` decorator, this is often handled automatically, but you can override or augment it.

```python
observe.input({"query": "hello"})
observe.output("world")
```

#### Status (`observe.status`, `observe.fail`)

Mark spans as successful or failed.

```python
observe.status("ok")
# or
observe.fail(Exception("Something went wrong"))
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
