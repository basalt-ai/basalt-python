# Manual Tracing

Learn how to create custom traces using the unified `start_observe` and `observe` interfaces.

## Overview

Basalt provides two main observability APIs:

- **`start_observe`**: Creates root spans with identity, experiment, and evaluator configuration
- **`observe`**: Creates nested spans with different kinds (generation, retrieval, tool, etc.)

Both work as:

- **Decorator**: Apply to functions for automatic span creation
- **Context Manager**: Fine-grained control within code blocks

## Quick Start

### Root Span with start_observe

Every trace must begin with a root span created using `start_observe`:

```python
from basalt.observability import start_observe, observe

@start_observe(
    feature_slug="data-processing",
    name="Data Processing Workflow",
    identity={
        "organization": {"id": "123", "name": "ACME"},
        "user": {"id": "456", "name": "John Doe"}
    },
    metadata={"version": "v2", "environment": "production"}
)
def process_data(data):
    # Identity and metadata automatically propagate to child spans
    prepare_data(data)
    transform_data(data)
    return result

@observe(name="Data Preparation")
def prepare_data(data):
    # This is a child span - inherits identity from root
    pass
```

### Using as a Decorator

```python
from basalt.observability import observe

@observe(name="Data Processing", kind="function")
def process_data(data):
    # Function automatically wrapped in a span
    return transform(data)
```

### Using as a Context Manager

```python
from basalt.observability import start_observe


def process_request():
    with start_observe(
            feature_slug="request-processing",
            name="Request Processing",
            identity={
                "organization": {"id": "123", "name": "ACME"},
                "user": {"id": "456", "name": "John Doe"}
            }
    ):
        observe.set_input({"data": "..."})
        result = do_work()
        observe.set_output(result)
        return result
```

## The start_observe and observe Interfaces

### start_observe: Root Span Entry Point

Use `start_observe` to create the root span of your trace. This is required as the entry point and supports:

```python
from basalt.observability import start_observe

# With identity tracking
@start_observe(
    feature_slug="main-workflow",
    name="Main Workflow",
    identity={
        "organization": {"id": "123", "name": "ACME"},
        "user": {"id": "456", "name": "John Doe"}
    },
    metadata={"environment": "production"}
)
def main_workflow(data):
    return process(data)

# With experiment tracking
@start_observe(
    feature_slug="ab-test",
    name="A/B Test",
    experiment={"id": "exp_001", "name": "Model Comparison", "variant": "model_a"},
    identity={
        "organization": {"id": "123", "name": "ACME"},
        "user": {"id": "456", "name": "John Doe"}
    }
)
def run_experiment():
    return results

# Context manager form
with start_observe(
    feature_slug="batch-job",
    name="Batch Job",
    identity={
        "organization": {"id": "123", "name": "ACME"},
        "user": {"id": "456", "name": "John Doe"}
    }
):
    # Your code here
    pass
```

**Key `start_observe` parameters:**

- `feature_slug` (required): A unique identifier for the feature or workflow being traced
- `name` (required): Span name (defaults to function name if used as decorator)
- `identity`: Dict with `user` and/or `organization` keys for tracking
- `experiment`: Dict with `id`, `name`, and `variant` for A/B testing
- `evaluate_config`: Evaluator configuration for the root span
- `metadata`: Custom key-value pairs

### observe: Nested Spans

Use `observe` for nested operations within a root span:

```python
from basalt.observability import observe, ObserveKind

@observe(name="Custom Operation", kind=ObserveKind.FUNCTION, metadata={"version": "v2"})
def my_operation(data):
    return result
```

### Span Kinds

Use the `kind` parameter to specify the type of operation:

```python
from basalt.observability import observe, ObserveKind

# Tool call
@observe(kind=ObserveKind.TOOL, name="Search Database")
def search_db(query):
    return results

# Function call
@observe(kind=ObserveKind.FUNCTION, name="Calculate Total")
def calculate(items):
    return sum(items)

# Event
@observe(kind=ObserveKind.EVENT, name="User Login")
def log_user_login(user_id):
    pass
```

Or use string values:

```python
@observe(kind="tool", name="Search Database")
def search_db(query):
    return results
```

### Nested Spans

Decorators automatically create parent-child relationships. Always start with `start_observe` as the root:

```python
from basalt.observability import start_observe, observe

@start_observe(
    feature_slug="main-workflow",
    name="Main Workflow",
    identity={
        "organization": {"id": "123", "name": "ACME"},
        "user": {"id": "456", "name": "John Doe"}
    }
)
def main_workflow():
    # Root span created here
    prepare_data()  # Creates child span
    process_data()  # Creates child span
    return result

@observe(name="Data Preparation")
def prepare_data():
    # Child span - inherits identity from root
    pass

@observe(name="Data Processing")
def process_data():
    # Another child span
    pass
```

## Context Manager Usage

### Working with Span Handles

When using `observe` as a context manager, you get a span handle with helpful methods:

```python
from basalt.observability import observe

with observe(name="API Request") as span:
    # Add metadata
    span.set_attribute("request_id", "abc123")

    # Set input
    span.set_input({"endpoint": "/api/users", "method": "GET"})

    # Do work
    result = fetch_data()

    # Set output
    span.set_output(result)
```

### Span Handle Methods

**Data Methods:**

- `set_input(data)` - Set input payload
- `set_output(data)` - Set output payload
- `set_attribute(key, value)` - Add metadata

**Status Methods:**

- `set_status(status, message)` - Set span status ("ok", "error", or "unset")
- `record_exception(exception)` - Record an exception

**Event Methods:**

- `add_event(name, attributes)` - Add a timestamped event

**Event Methods:**

- `add_event(name, attributes)` - Add a timestamped event

**Example with Error Handling:**

```python
from basalt.observability import observe

with observe(name="Database Query") as span:
    span.set_input({"query": "SELECT * FROM users"})

    try:
        result = execute_query()
        span.set_output(result)
        span.set_status("ok")
    except Exception as e:
        span.record_exception(e)
        span.set_status("error", str(e))
        raise
```

## Static Methods

The `observe` class provides static methods for working with the current span:

### Metadata Management

```python
from basalt.observability import observe

def process_data():
    # Add metadata to current span
    observe.metadata({"region": "us-west", "version": "v2"})

    # Or using kwargs
    observe.metadata(region="us-west", version="v2")

    # Merge metadata (updates existing keys)
    observe.update_metadata({"status": "processing"})
```

## Decorator vs Context Manager

Both decorators and context managers create spans, but they have different use cases and trade-offs.

### Quick Comparison

| Feature                   | Decorator                    | Context Manager                           |
| ------------------------- | ---------------------------- | ----------------------------------------- |
| **Syntax**                | `@observe(name="...")`       | `with observe(name="...") as span:`       |
| **Automatic I/O capture** | Yes (function args/return)   | No (manual with `span.set_input/output`)  |
| **Granular control**      | No                           | Yes (span handle methods)                 |
| **Best for**              | Function-level tracing       | Fine-grained control, multiple operations |

### When to Use Decorators

Use decorators when:

- You want to trace entire function execution
- Input/output capture from function args/return is sufficient
- You prefer clean, declarative code
- You don't need mid-function span control

```python
from basalt.observability import observe, ObserveKind

@observe(name="Data Processing", kind=ObserveKind.FUNCTION)
def process_data(data: dict) -> dict:
    # Function args automatically captured as input
    # Return value automatically captured as output
    result = transform(data)
    return result
```

### When to Use Context Managers

Use context managers when:

- You need granular control over span lifecycle
- You want to set attributes during execution
- You're tracing code blocks, not entire functions

```python
from basalt.observability import observe

def process_request(request_id: str):
    with observe(name="Request Processing") as span:
        # Fine-grained control
        span.set_input({"request_id": request_id})

        validate(request_id)

        result = process()

        # Set output
        span.set_output({"result": result, "status": "success"})
```

### Combining Both Patterns

You can mix decorators and context managers:

```python
from basalt.observability import start_observe, observe

@start_observe(feature_slug="main-workflow", name="Main Workflow")
def main_workflow(data):
    # Root span via decorator
    prepare_data(data)

    # Fine-grained control for critical section
    with observe(name="Critical Processing") as span:
        span.set_metadata({"priority": "high"})
        result = complex_operation(data)
        span.set_output(result)

    return result

@observe(name="Data Prep")
def prepare_data(data):
    # Simple decorator for straightforward function
    return clean(data)
```

### Span Handle Methods

When using context managers, the span handle provides these methods:

**Setting Data:**

- `set_input(data)` - Set input payload
- `set_output(data)` - Set output payload
- `set_metadata(dict)` - Set metadata

**Identity & Experiments:**

- `set_identity(identity)` - Set user/org identity using dict format: `{"user": {"id": "...", "name": "..."}, "organization": {"id": "...", "name": "..."}}`
- `add_evaluator(slug)` - Attach single evaluator
- `add_evaluators(*slugs)` - Attach multiple evaluators

**LLM-Specific (when `kind=ObserveKind.GENERATION`):**

- `set_model(model)` - Set LLM model name
- `set_prompt(prompt)` - Set prompt text
- `set_completion(completion)` - Set completion text
- `set_tokens(input=..., output=...)` - Set token counts

### Prompt Context Managers

Prompts fetched via `basalt.prompts.get_sync()` can also be used as context managers:

```python
from basalt import Basalt

basalt = Basalt(api_key="your-api-key")

# Context manager: creates span that scopes nested LLM calls
with basalt.prompts.get_sync("qa-prompt", variables={"context": "..."}) as prompt:
    # Auto-instrumented LLM calls nest under this span
    response = openai.chat.completions.create(
        model=prompt.model.model,
        messages=[{"role": "user", "content": prompt.text}]
    )

# Imperative: creates immediate span
prompt = basalt.prompts.get_sync("qa-prompt", variables={"context": "..."})
# LLM calls create separate spans
```

See [Prompts Documentation](03-prompts.md#observability-integration) for more details.

### Automatic Prompt Context Injection for Auto-Instrumented Spans

**NEW**: Basalt automatically injects prompt context into auto-instrumented LLM spans (OpenAI, Anthropic, Gemini, etc.) when they are called within a prompt context manager. This eliminates the need for manual prompt attribute setting.

```python
from basalt import Basalt
from basalt.observability import start_observe
from openai import OpenAI

basalt = Basalt(api_key="your-api-key")
openai_client = OpenAI(api_key="your-openai-key")

# Root trace
with start_observe(feature_slug="support-ticket", name="process_request"):
    # Fetch prompt with context manager
    with basalt.prompts.get_sync("joke-analyzer", variables={"jokeText": "..."}) as prompt:
        # Auto-instrumented OpenAI call automatically receives prompt attributes
        response = openai_client.chat.completions.create(
            model=prompt.model.model,
            messages=[{"role": "user", "content": prompt.text}]
        )
        # The OpenAI span will automatically have:
        # - basalt.prompt.slug = "joke-analyzer"
        # - basalt.prompt.version = (prompt version)
        # - basalt.prompt.model.provider = "openai"
        # - basalt.prompt.model.model = "gpt-4"
        # - basalt.prompt.variables = {"jokeText": "..."}
        # - basalt.prompt.from_cache = true/false
```

**How it works:**
- When you use a prompt context manager (`with prompts.get_sync(...)`), it sets an internal context variable
- Auto-instrumented spans (OpenAI, Gemini, Anthropic, etc.) automatically detect this context
- Prompt attributes are injected into the span without any manual intervention

**Supported providers:**
- OpenAI
- Anthropic (Claude)
- Google Gemini
- AWS Bedrock
- Mistral
- Any other auto-instrumented LLM provider

**Benefits:**
- Zero boilerplate - no manual attribute setting required
- Consistent across all providers
- Works with nested prompt contexts (inner context wins)
- Thread-safe and async-compatible

## Best Practices

### Set Input and Output

Always set input and output for better observability:

```python
@observe(name="Calculate Discount")
def calculate_discount(user_tier, amount):
    with observe(name="Discount Calculation") as span:
        span.set_input({"user_tier": user_tier, "amount": amount})

        discount = compute_discount(user_tier, amount)

        span.set_output({"discount": discount})
        return discount
```
