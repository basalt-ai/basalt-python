# User & Organization Tracking

Track user and organization identity across your observability traces to understand usage patterns.

## Core Concepts

### TraceIdentity

The `TraceIdentity` data structure represents a user or organization:

```python
@dataclass(frozen=True, slots=True)
class TraceIdentity:
    id: str                      # Required unique identifier
    name: str | None = None      # Optional display name
```

### Semantic Conventions

User and organization data is stored using standardized span attributes:

**User Attributes:**
- `basalt.user.id` - Unique user identifier (required)
- `basalt.user.name` - User display name (optional)

**Organization Attributes:**
- `basalt.organization.id` - Unique organization identifier (required)
- `basalt.organization.name` - Organization display name (optional)

## Quick Start

### Setting Identity on Root Span

The recommended approach is to set identity using the `identity` parameter on `start_observe`:

```python
from basalt.observability import start_observe

# With both user and organization
@start_observe(
    name="my.operation",
    identity={
        "organization": {"id": "123", "name": "ACME"},
        "user": {"id": "456", "name": "John Doe"}
    }
)
def my_operation():
    # Identity automatically propagates to all child spans
    pass

# With just user ID (simple form)
@start_observe(
    name="process",
    identity={
        "organization": {"id": "123", "name": "ACME"},
        "user": {"id": "456", "name": "John Doe"}
    }
)
def process():
    pass

# Context manager form
with start_observe(
    name="batch_job",
    identity={
        "organization": {"id": "123", "name": "ACME"},
        "user": {"id": "456", "name": "John Doe"}
    }
):
    # Your code here
    pass
```

### Dynamic Identity Resolution

Use a callable for identity when values depend on function arguments:

```python
from basalt.observability import start_observe

@start_observe(
    name="handle_request",
    identity=lambda user_id, org_id, **kwargs: {
        "organization": {"id": "123", "name": "ACME"},
        "user": {"id": "456", "name": "John Doe"}
    }
)
def handle_request(user_id: str, org_id: str, data: dict):
    # Identity resolved from function arguments
    return process(data)
```

### Setting Identity Dynamically

Use `observe.identify()` to set identity at runtime:

```python
from basalt.observability import start_observe, observe

@start_observe(name="api_handler")
def handle_api_request(auth_token):
    # Extract identity from auth token
    user_data = verify_token(auth_token)

    # Set identity dynamically
    observe.identify(
        user={"id": "456", "name": "John Doe"},
        organization={"id": "123", "name": "ACME"}
    )

    # Identity now propagates to all subsequent operations
    process_request()
```

## Setting Identity

### Method 1: start_observe identity Parameter (Recommended)

Set identity on the root span using the `identity` parameter:

```python
from basalt.observability import start_observe

@start_observe(
    name="my.operation",
    identity={
        "organization": {"id": "123", "name": "ACME"},
        "user": {"id": "456", "name": "John Doe"}
    }
)
def my_operation():
    # Identity set on root and propagates to children
    pass
```

### Method 2: observe.identify() Static Method

Set identity dynamically from anywhere in your code:

```python
from basalt.observability import observe

def handle_request(auth_data):
    # Extract identity from runtime data
    observe.identify(
        user={"id": "456", "name": "John Doe"},
        organization={"id": "123", "name": "ACME"}
    )
    # Identity propagates to all spans in this trace
```

### Method 3: SpanHandle Methods

Use span handles for fine-grained control:

```python
from basalt.observability import start_observe, observe

with start_observe(name="my.operation") as span:
    span.set_user("456", name="John Doe")
    span.set_organization("123", name="ACME")
```

### Method 4: Callable Identity Resolver

For decorator-based identity resolution from function arguments:

```python
from basalt.observability import start_observe

@start_observe(
    name="process_user_data",
    identity=lambda user_id, **kwargs: {
        "organization": {"id": "123", "name": "ACME"},
        "user": {"id": "456", "name": "John Doe"}
    }
)
def process_user_data(user_id: str, data: dict):
    # user_id automatically extracted and set as identity
    pass
```

### Method 4: set_trace_user() Helper

Set user on the current active span:

```python
from basalt.observability import set_trace_user

set_trace_user("456", name="John Doe")
```

## Setting Organization IDs

### Method 1: SpanHandle.set_organization()

Directly set organization on a span handle:

```python
from basalt.observability import trace_span

with trace_span("my.operation") as span:
    span.set_organization("123", name="ACME")
```

### Method 2: Context Manager Parameter

Pass organization when creating the span:

```python
from basalt.observability import trace_span

with trace_span(
    "my.operation",
    organization={"id": "123", "name": "ACME"}
) as span:
    # Organization is automatically set
    pass
```


### Method 3: Decorator Parameter

Use decorators with organization parameter:

```python
from basalt.observability import observe_generation

@observe_generation(
    organization={"id": "123", "name": "ACME"}
)
def call_llm(prompt: str) -> str:
    return model.generate(prompt)
```

### Method 4: set_trace_organization() Helper

Set organization on the current active span:

```python
from basalt.observability import set_trace_organization

set_trace_organization("123", name="ACME")
```

## Context Managers with User/Org

All specialized context managers support user and organization parameters:

### trace_span()

```python
from basalt.observability import trace_span

with trace_span(
    "my.operation",
    user={"id": "456", "name": "John Doe"},
    organization={"id": "123", "name": "ACME"}
) as span:
    # Both user and org are set
    pass
```


### trace_generation()

For LLM operations:

```python
from basalt.observability import trace_generation

with trace_generation(
    "llm.generate",
    user={"id": "456", "name": "John Doe"},
    organization={"id": "123", "name": "ACME"}
) as span:
    span.set_model("gpt-4")
    span.set_prompt("Hello")
    result = call_llm()
    span.set_completion(result)
```


### trace_retrieval()

For vector database operations:

```python
from basalt.observability import trace_retrieval

with trace_retrieval(
    "vector_db.search",
    user={"id": "456", "name": "John Doe"},
    organization={"id": "123", "name": "ACME"}
) as span:
    span.set_query("What is observability?")
    span.set_top_k(10)
    results = search_db(query)
    span.set_results_count(len(results))
```


### trace_function(), trace_tool(), trace_event()

All other specialized context managers also support user and organization:

```python
from basalt.observability import trace_function, trace_tool, trace_event

# Function execution
with trace_function(
    "my.func",
    user={"id": "456", "name": "John Doe"},
    organization={"id": "123", "name": "ACME"}
) as span:
    pass

# Tool execution
with trace_tool(
    "my.tool",
    user={"id": "456", "name": "John Doe"},
    organization={"id": "123", "name": "ACME"}
) as span:
    pass

# Event logging
with trace_event(
    "my.event",
    user={"id": "456", "name": "John Doe"},
    organization={"id": "123", "name": "ACME"}
) as span:
    pass
```


## Decorators with User/Org

All specialized decorators support user and organization parameters:

### observe()

```python
from basalt.observability import observe, ObserveKind

@observe(
    ObserveKind.GENERATION,
    user={"id": "456", "name": "John Doe"},
    organization={"id": "123", "name": "ACME"}
)
def process_data(query: str) -> str:
    return query.upper()
```


### observe_generation()

```python
from basalt.observability import observe_generation

@observe_generation(
    user={"id": "456", "name": "John Doe"},
    organization={"id": "123", "name": "ACME"}
)
def call_llm(prompt: str) -> str:
    return model.generate(prompt)
```


### observe_retrieval(), observe_function(), observe_tool(), observe_event()

All specialized decorators support the same pattern:

```python
from basalt.observability import (
    observe_retrieval,
    observe_function,
    observe_tool,
    observe_event
)

@observe_retrieval(
    user={"id": "456", "name": "John Doe"},
    organization={"id": "123", "name": "ACME"}
)
def search(query: str):
    pass

@observe_function(
    user={"id": "456", "name": "John Doe"},
    organization={"id": "123", "name": "ACME"}
)
def process(data: str):
    pass

@observe_tool(
    user={"id": "456", "name": "John Doe"},
    organization={"id": "123", "name": "ACME"}
)
def execute_tool(params: dict):
    pass

@observe_event(
    user={"id": "456", "name": "John Doe"},
    organization={"id": "123", "name": "ACME"}
)
def log_event(event: str):
    pass
```

## Dynamic User/Org Resolution

### Using Callable Resolvers

When using decorators, you can provide callable resolvers that extract user/org from function arguments:

```python
from basalt.observability import observe_generation, TraceIdentity

@observe_generation(
    user=lambda bound: TraceIdentity(
        id="456",
        name="John Doe"
    ),
    organization=lambda bound: TraceIdentity(
        id="123",
        name="ACME"
    )
)
def process(user_id: str, username: str, prompt: str) -> str:
    return model.generate(prompt)

# When called:
result = process(
    user_id="456",
    username="John Doe",
    prompt="Hello"
)
# User "456" (John Doe) is automatically attached to the span
```


### Dynamic Organization

```python
@observe_generation(
    user=lambda bound: {
        "id": "456",
        "name": "John Doe"
    },
    organization=lambda bound: {
        "id": "123",
        "name": "ACME"
    }
)
def handle_request(org_id: str, org_name: str, query: str) -> str:
    return process_query(query)
```

## Context Propagation

User and organization automatically propagate to child spans through OpenTelemetry context:

```python
from basalt.observability import trace_span

with trace_span(
    "parent.operation",
    user={"id": "456", "name": "John Doe"},
    organization={"id": "123", "name": "ACME"}
):
    # Child span automatically inherits user and org
    with trace_span("child.operation") as child:
        # child span has 456 and 123 automatically
        pass
```

### Automatic Propagation to Auto-Instrumented Spans

User and organization also propagate to auto-instrumented LLM calls:

```python
from basalt.observability import trace_span
from openai import OpenAI

client = OpenAI(api_key="your-key")

with trace_span(
    "user.request",
    user={"id": "456", "name": "John Doe"},
    organization={"id": "123", "name": "ACME"}
):
    # This auto-instrumented OpenAI call inherits user and org
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}]
    )
```


## Complete Example

### Real-World Usage Pattern

```python
from basalt import Basalt, TelemetryConfig
from basalt.observability import trace_span, trace_generation, trace_retrieval

# Initialize Basalt
telemetry = TelemetryConfig(
    service_name="customer-support",
    environment="production",
    enable_llm_instrumentation=True,
)
basalt = Basalt(api_key="your-api-key", telemetry_config=telemetry)

def handle_customer_query(user_id: str, user_name: str, org_id: str, org_name: str, query: str):
    """Handle a customer support query with full tracking"""

    with trace_span(
        "support.handle_query",
        user={"id": "456", "name": "John Doe"},
        organization={"id": "123", "name": "ACME"}
    ) as root_span:
        root_span.set_input({"query": query})

        # Step 1: Search knowledge base
        # User and org automatically propagate to this span
        with trace_retrieval("kb.search") as search_span:
            search_span.set_query(query)
            search_span.set_top_k(5)

            results = search_knowledge_base(query)
            search_span.set_results_count(len(results))

        # Step 2: Generate response
        # User and org automatically propagate here too
        with trace_generation("llm.generate_response") as gen_span:
            gen_span.set_model("gpt-4")

            context = "\n".join([r["content"] for r in results])
            prompt = f"Context: {context}\n\nQuery: {query}\n\nAnswer:"

            gen_span.set_prompt(prompt)

            response = call_llm(prompt)
            gen_span.set_completion(response)

        root_span.set_output({"response": response})
        return response

# Call the function
response = handle_customer_query(
    user_id="456",
    user_name="John Doe",
    org_id="123",
    org_name="ACME",
    query="How do I reset my password?"
)

basalt.shutdown()
```


## Input Formats

User and organization accept multiple input formats:

### TraceIdentity Object

```python
from basalt.observability import TraceIdentity, trace_span

user = TraceIdentity(id="456", name="John Doe")
org = TraceIdentity(id="123", name="ACME")

with trace_span("my.op", user=user, organization=org) as span:
    pass
```

### Dictionary

```python
with trace_span(
    "my.op",
    user={"id": "456", "name": "John Doe"},
    organization={"id": "123", "name": "ACME"}
) as span:
    pass
```

### Callable (Decorators Only)

```python
@observe_generation(
    user=lambda bound: {"id": bound.arguments.get("user_id")}
)
def process(user_id: str) -> str:
    pass
```
