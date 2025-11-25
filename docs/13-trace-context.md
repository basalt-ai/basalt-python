# Trace Context

Learn how Basalt uses OpenTelemetry context propagation to pass metadata across spans and services in distributed systems.

## What is Trace Context?

**Trace Context** is OpenTelemetry's mechanism for propagating metadata through your application and across service boundaries. In Basalt, trace context manages:

- **User Identity**: User ID and name
- **Organization Identity**: Organization ID and name
- **Experiment Metadata**: Experiment tracking information
- **Evaluator Configuration**: Which evaluators to attach to spans
- **Custom Metadata**: Application-specific attributes

The context automatically flows from parent spans to child spans, ensuring consistent tracking throughout distributed operations.

## Core Concepts

### OpenTelemetry Context

OpenTelemetry uses **context** as a key-value store that propagates through your application:

```python
from opentelemetry import context

# Set a value in context
token = context.attach(context.set_value("key", "value"))

# Get a value from context
value = context.get_value("key")

# Detach when done
context.detach(token)
```

Basalt builds on this foundation to automatically propagate observability metadata.

### Context Keys

Basalt uses these standardized context keys:

```python
# User and organization identity
USER_CONTEXT_KEY = "basalt.context.user"
ORGANIZATION_CONTEXT_KEY = "basalt.context.organization"

# Evaluator configuration
EVALUATOR_CONTEXT_KEY = "basalt.context.evaluators"
EVALUATOR_CONFIG_CONTEXT_KEY = "basalt.context.evaluator_config"
EVALUATOR_METADATA_CONTEXT_KEY = "basalt.context.evaluator_metadata"
```

### Data Structures

**TraceIdentity** - User or organization identity:

```python
@dataclass(frozen=True, slots=True)
class TraceIdentity:
    id: str
    name: str | None = None
```

**TraceExperiment** - Experiment metadata:

```python
@dataclass(frozen=True, slots=True)
class TraceExperiment:
    id: str
    name: str | None = None
    feature_slug: str | None = None
```

**TraceContextConfig** - Default span attributes:

```python
@dataclass(slots=True)
class TraceContextConfig:
    experiment: TraceExperiment | Mapping[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    evaluators: list[Any] | None = None
```

## How Context Propagation Works

### Automatic Propagation

When you set user or organization on a span, it automatically propagates to all child spans:

```python
from basalt.observability import trace_span

with trace_span(
    "parent",
    user={"id": "user-123"},
    organization={"id": "org-456"}
):
    # Child span automatically inherits user and org
    with trace_span("child"):
        # This span has user-123 and org-456
        pass

    # Auto-instrumented spans also inherit
    response = openai_client.chat.completions.create(...)
    # OpenAI span has user-123 and org-456
```


### Context Attachment Mechanism

Basalt uses context tokens to manage propagation:

```python
# Simplified version of what happens internally
from opentelemetry.context import attach, set_value, detach

# Attach user to context
user_identity = TraceIdentity(id="user-123", name="Alice")
token = attach(set_value(USER_CONTEXT_KEY, user_identity))

try:
    # User is now in context for all child spans
    create_child_span()
finally:
    # Clean up when done
    detach(token)
```

### Processor-Based Application

The `BasaltContextProcessor` automatically reads context and applies it to spans:

```python
class BasaltContextProcessor(SpanProcessor):
    def on_start(self, span: Span, parent_context: Any | None = None) -> None:
        # Read user from context
        user = context.get_value(USER_CONTEXT_KEY, parent_context)
        if isinstance(user, TraceIdentity):
            span.set_attribute("basalt.user.id", user.id)
            if user.name:
                span.set_attribute("basalt.user.name", user.name)

        # Read organization from context
        org = context.get_value(ORGANIZATION_CONTEXT_KEY, parent_context)
        if isinstance(org, TraceIdentity):
            span.set_attribute("basalt.organization.id", org.id)
            if org.name:
                span.set_attribute("basalt.organization.name", org.name)
```

## Extracting Trace Context

### Get User from Context

```python
from basalt.observability.trace_context import get_context_user

# Returns TraceIdentity if set, None otherwise
user = get_context_user()
if user:
    print(f"Current user: {user.id} ({user.name})")
```

### Get Organization from Context

```python
from basalt.observability.trace_context import get_context_organization

org = get_context_organization()
if org:
    print(f"Current org: {org.id} ({org.name})")
```

### Get Current Span

```python
from basalt.observability import current_span

# Get the active OpenTelemetry span
span = current_span()
if span:
    span.set_attribute("custom.key", "value")
```

### Get Current SpanHandle

```python
from basalt.observability import current_span_handle

# Get a SpanHandle wrapper for easier manipulation
handle = current_span_handle()
if handle:
    handle.set_user("user-123")
    handle.add_evaluator("quality-check")
```


## Injecting Trace Context

### Apply User to Span

```python
from basalt.observability.trace_context import apply_user_from_context
from opentelemetry import trace

# Get or create a span
span = trace.get_current_span()

# Apply user from context (or explicit value)
apply_user_from_context(span)

# Or with explicit user
apply_user_from_context(span, user={"id": "user-123"})
```

### Apply Organization to Span

```python
from basalt.observability.trace_context import apply_organization_from_context

span = trace.get_current_span()

# Apply organization from context
apply_organization_from_context(span)

# Or with explicit organization
apply_organization_from_context(span, organization={"id": "org-456"})
```

## Global Trace Defaults

### Configuring Defaults

Set default attributes applied to all new spans:

```python
from basalt.observability import configure_trace_defaults, TraceExperiment

configure_trace_defaults(
    experiment=TraceExperiment(
        id="exp-default",
        name="Default Experiment"
    ),
    metadata={
        "environment": "production",
        "version": "2.0"
    },
    evaluators=["safety-check", "quality-eval"]
)
```


### Getting Current Defaults

```python
from basalt.observability import current_trace_defaults

defaults = current_trace_defaults()
print(f"Default experiment: {defaults.experiment.id}")
print(f"Default metadata: {defaults.metadata}")
print(f"Default evaluators: {defaults.evaluators}")
```


### Setting Defaults

```python
from basalt.observability.trace_context import set_trace_defaults, TraceContextConfig

config = TraceContextConfig(
    experiment={"id": "exp-123", "name": "My Experiment"},
    metadata={"region": "us-west"},
    evaluators=["eval-1", "eval-2"]
)

set_trace_defaults(config)
```

## Evaluator Context Propagation

### Understanding Evaluator Propagation

Evaluators have **two modes of attachment**:

1. **Propagating** - Evaluators flow through OpenTelemetry context to child spans
2. **Non-Propagating** - Evaluators attach only to specific spans

**Propagating methods:**

```python
from basalt.observability import evaluator, trace_span

@evaluator("parent-eval")
def parent_operation():
    with trace_span("parent.span"):  # Gets "parent-eval"
        # Child inherits "parent-eval" via propagation
        child_operation()

def child_operation():
    with trace_span("child.span"):  # Also has "parent-eval"
        pass
```

The `@evaluator` decorator, `with_evaluators()`, and `attach_evaluator()` context manager all use propagation.

**Non-propagating methods:**

```python
from basalt.observability import trace_span

with trace_span("parent") as parent:
    parent.add_evaluator("parent-only")  # Only on parent

    with trace_span("child") as child:
        # Child does NOT have "parent-only"
        child.add_evaluator("child-only")  # Only on child
```

Direct span methods like `span.add_evaluator()` do NOT propagate.

### with_evaluators Context Manager (Advanced)

The `with_evaluators()` context manager is a low-level API for explicit propagation:

```python
from basalt.observability.context_managers import with_evaluators
from basalt.observability import EvaluationConfig, trace_span

with with_evaluators(
    evaluators=["eval-1", "eval-2"],
    config=EvaluationConfig(sample_rate=0.5),
    metadata={"source": "api"}
):
    # Any span created in this scope gets these evaluators
    with trace_span("parent"):  # Has eval-1, eval-2
        with trace_span("child"):  # Also has eval-1, eval-2 (propagated)
            pass
```

**When to use**: Building custom instrumentation, middleware, or need explicit control over propagation scope.

**File reference:** `basalt/observability/context_managers.py:109-159`

## Helper Functions

### update_current_span()

Update the active span with input/output and evaluators:

```python
from basalt.observability import update_current_span

# Inside a traced function
update_current_span(
    input_payload={"query": "hello"},
    output_payload={"result": "world"},
    variables={"user": "alice"},
    evaluators=["quality-check"]
)
```

### set_trace_user() and set_trace_organization()

Set user/org on current span AND propagate to children:

```python
from basalt.observability import set_trace_user, set_trace_organization

# Set user on current span + propagate to children
set_trace_user("user-123", name="Alice")

# Set organization on current span + propagate to children
set_trace_organization("org-456", name="Acme Corp")
```

## Distributed Systems Usage

### Parent-Child Propagation

Context propagates naturally through nested operations:

```python
from basalt.observability import trace_span

def handle_request(user_id: str, org_id: str):
    with trace_span(
        "api.request",
        user={"id": user_id},
        organization={"id": org_id}
    ):
        # All these inherit user and org
        fetch_user_data()
        process_business_logic()
        save_results()

def fetch_user_data():
    with trace_span("db.fetch_user"):
        # Has user and org automatically
        query_database()

def process_business_logic():
    with trace_span("logic.process"):
        # Has user and org automatically
        call_llm()

def call_llm():
    with trace_generation("llm.call"):
        # Has user and org automatically
        pass
```

### Cross-Service Propagation

For microservices, use OpenTelemetry's built-in context propagation:

```python
# Service A
from opentelemetry import trace
from opentelemetry.propagate import inject

def call_service_b():
    with trace_span("service_a.call_b", user={"id": "user-123"}):
        # Prepare headers for HTTP request
        headers = {}
        inject(headers)  # Injects trace context into headers

        # Make HTTP request
        response = requests.post(
            "http://service-b/endpoint",
            headers=headers
        )
```

```python
# Service B
from opentelemetry.propagate import extract

@app.route("/endpoint", methods=["POST"])
def handle_request():
    # Extract trace context from headers
    context = extract(request.headers)

    # Use the extracted context
    with trace_span("service_b.handle", context=context):
        # This span is part of the same trace from Service A
        # and inherits user-123
        process_request()
```

## Complete Example

### Multi-Layer Application with Context Propagation

```python
from basalt import Basalt, TelemetryConfig
from basalt.observability import (
    trace_span,
    trace_generation,
    trace_retrieval,
    configure_trace_defaults
)

# Initialize Basalt
telemetry = TelemetryConfig(
    service_name="multi-layer-app",
    environment="production",
    enable_llm_instrumentation=True,
)
basalt = Basalt(api_key="your-api-key", telemetry_config=telemetry)

# Set global defaults
configure_trace_defaults(
    metadata={"app_version": "2.0.0"},
)

def api_handler(user_id: str, user_name: str, org_id: str, query: str):
    """API entry point - sets user and org context"""

    with trace_span(
        "api.handle_query",
        user={"id": user_id, "name": user_name},
        organization={"id": org_id, "name": "Customer Org"}
    ) as root:
        root.set_input({"query": query})

        # Context automatically propagates to all layers
        result = service_layer(query)

        root.set_output({"result": result})
        return result

def service_layer(query: str):
    """Service layer - inherits user/org from parent"""

    with trace_span("service.process_query") as span:
        # User and org are already set via context!

        # Retrieve context
        context_docs = data_layer(query)

        # Generate response
        response = llm_layer(query, context_docs)

        return response

def data_layer(query: str):
    """Data layer - inherits user/org from parent"""

    with trace_retrieval("data.search") as span:
        # User and org are already set!
        span.set_query(query)
        span.set_top_k(5)

        # Simulate database search
        results = search_database(query)

        span.set_results_count(len(results))
        return results

def llm_layer(query: str, context: list):
    """LLM layer - inherits user/org from parent"""

    with trace_generation("llm.generate") as span:
        # User and org are already set!
        span.set_model("gpt-4")

        prompt = f"Context: {context}\n\nQuery: {query}"
        span.set_prompt(prompt)

        response = call_llm(prompt)
        span.set_completion(response)

        return response

# Call the API
result = api_handler(
    user_id="user-123",
    user_name="Alice Johnson",
    org_id="org-456",
    query="What is observability?"
)

basalt.shutdown()
```

**Result**: All spans in the trace have:
- `basalt.user.id = "user-123"`
- `basalt.user.name = "Alice Johnson"`
- `basalt.organization.id = "org-456"`
- `basalt.organization.name = "Customer Org"`
- Metadata: `app_version = "2.0.0"`

## Context Validation

### Input Validation

Basalt validates context values before storing them:

```python
def _coerce_identity(payload: TraceIdentity | Mapping[str, Any] | None):
    """Validate and coerce user/org identity."""
    if payload is None or isinstance(payload, TraceIdentity):
        return payload

    if not isinstance(payload, Mapping):
        raise TypeError("Identity must be a mapping or TraceIdentity")

    identifier = payload.get("id")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("Identity requires a non-empty 'id'")

    name = payload.get("name")
    if name is not None and not isinstance(name, str):
        raise ValueError("Identity 'name' must be a string")

    return TraceIdentity(id=identifier, name=name)
```
