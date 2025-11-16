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

### Setting User on a Span

```python
from basalt.observability import trace_span

with trace_span("my.operation") as span:
    span.set_user("user-123", name="Alice Johnson")
    # Your operation here
```

### Setting Organization on a Span

```python
from basalt.observability import trace_span

with trace_span("my.operation") as span:
    span.set_organization("org-456", name="Acme Corp")
    # Your operation here
```

### Setting Both User and Organization

```python
from basalt.observability import trace_span

with trace_span("my.operation") as span:
    span.set_user("user-123", name="Alice Johnson")
    span.set_organization("org-456", name="Acme Corp")
    # Your operation here
```

## Setting User IDs

### Method 1: SpanHandle.set_user()

Directly set user on a span handle:

```python
from basalt.observability import trace_span

with trace_span("my.operation") as span:
    span.set_user("user-123", name="Alice Johnson")
```


### Method 2: Context Manager Parameter

Pass user when creating the span:

```python
from basalt.observability import trace_span

with trace_span(
    "my.operation",
    user={"id": "user-123", "name": "Alice"}
) as span:
    # User is automatically set
    pass
```


### Method 3: Decorator Parameter

Use decorators with user parameter:

```python
from basalt.observability import observe_generation

@observe_generation(
    user={"id": "user-123", "name": "Alice"}
)
def process_data(query: str) -> str:
    return query.upper()
```


### Method 4: set_trace_user() Helper

Set user on the current active span:

```python
from basalt.observability import set_trace_user

set_trace_user("user-123", name="Alice Johnson")
```

## Setting Organization IDs

### Method 1: SpanHandle.set_organization()

Directly set organization on a span handle:

```python
from basalt.observability import trace_span

with trace_span("my.operation") as span:
    span.set_organization("org-456", name="Acme Corp")
```

### Method 2: Context Manager Parameter

Pass organization when creating the span:

```python
from basalt.observability import trace_span

with trace_span(
    "my.operation",
    organization={"id": "org-456", "name": "Acme Corp"}
) as span:
    # Organization is automatically set
    pass
```


### Method 3: Decorator Parameter

Use decorators with organization parameter:

```python
from basalt.observability import observe_generation

@observe_generation(
    organization={"id": "org-456", "name": "Acme Corp"}
)
def call_llm(prompt: str) -> str:
    return model.generate(prompt)
```

### Method 4: set_trace_organization() Helper

Set organization on the current active span:

```python
from basalt.observability import set_trace_organization

set_trace_organization("org-456", name="Acme Corp")
```

## Context Managers with User/Org

All specialized context managers support user and organization parameters:

### trace_span()

```python
from basalt.observability import trace_span

with trace_span(
    "my.operation",
    user={"id": "user-123", "name": "Alice"},
    organization={"id": "org-456", "name": "Acme Corp"}
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
    user={"id": "user-123", "name": "Alice"},
    organization={"id": "org-456", "name": "Acme Corp"}
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
    user={"id": "user-123", "name": "Alice"},
    organization={"id": "org-456", "name": "Acme Corp"}
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
with trace_function("my.func", user={"id": "user-123"}) as span:
    pass

# Tool execution
with trace_tool("my.tool", user={"id": "user-123"}) as span:
    pass

# Event logging
with trace_event("my.event", user={"id": "user-123"}) as span:
    pass
```


## Decorators with User/Org

All specialized decorators support user and organization parameters:

### observe()

```python
from basalt.observability import observe, ObserveKind

@observe(
    ObserveKind.GENERATION,
    user={"id": "user-123", "name": "Alice"},
    organization={"id": "org-456", "name": "Acme Corp"}
)
def process_data(query: str) -> str:
    return query.upper()
```


### observe_generation()

```python
from basalt.observability import observe_generation

@observe_generation(
    user={"id": "user-123"},
    organization={"id": "org-456"}
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

@observe_retrieval(user={"id": "user-123"})
def search(query: str):
    pass

@observe_function(user={"id": "user-123"})
def process(data: str):
    pass

@observe_tool(user={"id": "user-123"})
def execute_tool(params: dict):
    pass

@observe_event(user={"id": "user-123"})
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
        id=bound.arguments.get("user_id"),
        name=bound.arguments.get("username")
    )
)
def process(user_id: str, username: str, prompt: str) -> str:
    return model.generate(prompt)

# When called:
result = process(
    user_id="user-123",
    username="Alice",
    prompt="Hello"
)
# User "user-123" (Alice) is automatically attached to the span
```


### Dynamic Organization

```python
@observe_generation(
    organization=lambda bound: {
        "id": bound.arguments.get("org_id"),
        "name": bound.arguments.get("org_name")
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
    user={"id": "user-123", "name": "Alice"},
    organization={"id": "org-456", "name": "Acme Corp"}
):
    # Child span automatically inherits user and org
    with trace_span("child.operation") as child:
        # child span has user-123 and org-456 automatically
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
    user={"id": "user-123"},
    organization={"id": "org-456"}
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
        user={"id": user_id, "name": user_name},
        organization={"id": org_id, "name": org_name}
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
    user_id="user-123",
    user_name="Alice Johnson",
    org_id="org-456",
    org_name="Acme Corp",
    query="How do I reset my password?"
)

basalt.shutdown()
```


## Input Formats

User and organization accept multiple input formats:

### TraceIdentity Object

```python
from basalt.observability import TraceIdentity, trace_span

user = TraceIdentity(id="user-123", name="Alice")
org = TraceIdentity(id="org-456", name="Acme Corp")

with trace_span("my.op", user=user, organization=org) as span:
    pass
```

### Dictionary

```python
with trace_span(
    "my.op",
    user={"id": "user-123", "name": "Alice"},
    organization={"id": "org-456", "name": "Acme Corp"}
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

### Validation

The SDK validates user and organization inputs:

```python
# Valid
TraceIdentity(id="user-123", name="Alice")  # ✓
{"id": "user-123", "name": "Alice"}         # ✓
{"id": "user-123"}                          # ✓ (name optional)

# Invalid
{"name": "Alice"}                           # ✗ (missing id)
{"id": ""}                                  # ✗ (empty id)
{"id": "user-123", "name": 123}             # ✗ (name not string)
```
