# Evaluators

Evaluators are a powerful system for attaching evaluation specifications to your LLM and application spans. They enable you to track which evaluations should be run on specific operations, helping you assess quality, safety, and performance of your AI applications.

## What are Evaluators?

**Evaluators** in Basalt are lightweight identifiers (slugs) that you attach to OpenTelemetry spans to indicate which evaluations should be performed. They work by:

- Attaching evaluator **slugs** (string identifiers like `"hallucination-check"`, `"quality-eval"`) to spans
- Optionally providing **configuration** (like sample rates) at the span level
- Optionally including **metadata** specific to each evaluator or shared across the span
- Optionally **propagating** through context to child spans (depending on the method used)

The system is flexible - you define evaluators by their slug identifiers, and choose whether they apply to specific spans or propagate to children.

## Core Concepts

### Evaluator Slugs

Evaluator slugs are string identifiers that represent specific evaluation types:

```python
# Common evaluator slug examples
"hallucination-check"     # Verify outputs don't contain false information
"quality-eval"            # Assess quality of responses
"answer-correctness"      # Verify answer accuracy
"toxicity"                # Safety evaluation
"consistency-check"       # Check for consistency across variants
"response-quality"        # General response quality
```

### EvaluationConfig

Type-safe configuration applied at the span level:

```python
from basalt.observability import EvaluationConfig

config = EvaluationConfig(
    sample_rate=0.5  # Evaluate 50% of calls (0.0 to 1.0)
)
```

### EvaluatorAttachment

Internal structure representing an evaluator with optional metadata:

```python
@dataclass(slots=True)
class EvaluatorAttachment:
    slug: str  # Required evaluator identifier
    metadata: Mapping[str, Any] | None = None  # Optional metadata
```

## Propagation Behavior

Evaluators can be attached in two ways:

### Propagating Evaluators (Affect Child Spans)

Some methods propagate evaluators through OpenTelemetry context to all child spans created within their scope:

| Method | Propagates? | Use Case |
|--------|-------------|----------|
| `@evaluator` decorator | ✅ Yes | Apply evaluators to all spans within a function |
| `with_evaluators()` | ✅ Yes | Low-level API for explicit propagation |
| `attach_evaluator()` context manager | ✅ Yes | Attach to specific span AND propagate |
| Global defaults (`configure_trace_defaults`) | ✅ Yes | Apply to all spans in application |

**Example - Propagating**:
```python
from basalt.observability import evaluator, trace_span

@evaluator("quality-check")
def process_request(query: str):
    with trace_span("parent"):  # Gets "quality-check"
        with trace_span("child"):  # Also gets "quality-check" (propagated)
            pass
```

### Non-Propagating Evaluators (Span-Specific Only)

These methods attach evaluators only to the specific span, without affecting children:

| Method | Propagates? | Use Case |
|--------|-------------|----------|
| `span.add_evaluator()` | ❌ No | Attach to one specific span only |
| `attach_evaluators_to_span()` | ❌ No | Direct attachment helper |
| `attach_evaluators_to_current_span()` | ❌ No | Attach to current span only |

**Example - Non-Propagating**:
```python
from basalt.observability import trace_span

with trace_span("parent") as parent_span:
    parent_span.add_evaluator("parent-eval")  # Only on parent

    with trace_span("child") as child_span:
        # Child does NOT have "parent-eval"
        child_span.add_evaluator("child-eval")  # Only on child
```

### Choosing the Right Approach

**Use propagating methods when:**
- You want to evaluate an entire workflow or function
- All operations within a scope should be evaluated together
- Using the `@evaluator` decorator on a function

**Use non-propagating methods when:**
- You need fine-grained control over which spans get evaluated
- Different steps in a workflow need different evaluators
- You want to avoid evaluator "pollution" of child spans

## Quick Start

### Method 1: Using the @evaluator Decorator

The easiest way to add evaluators is with the `@evaluator` decorator:

```python
from basalt.observability import observe_generation, evaluator

@evaluator("joke-quality")
@observe_generation(name="gemini.summarize_joke")
def summarize_joke_with_gemini(joke: str) -> str:
    return call_llm(joke)
```

### Method 2: Using Context Managers

For fine-grained control, use the `attach_evaluator` context manager:

```python
from basalt.observability import trace_generation, attach_evaluator

with trace_generation("my.llm") as llm_span:
    with attach_evaluator("hallucination-check", "quality-eval"):
        llm_span.set_model("gpt-4")
        llm_span.set_prompt("Tell me a joke")
        result = call_llm()
        llm_span.set_completion(result)
```

### Method 3: Direct Span Methods

Attach evaluators directly to span handles:

```python
from basalt.observability import trace_generation

with trace_generation("test.llm") as span:
    span.add_evaluator("llm-eval")
    span.set_model("gpt-4")
    span.set_prompt("Explain quantum computing")
    result = llm.generate("Explain quantum computing")
    span.set_completion(result)
```

## Using the @evaluator Decorator

The `@evaluator` decorator **propagates evaluators to all child spans** created within the decorated function.

### Basic Usage

```python
@evaluator("joke-quality")
@observe_generation(name="gemini.summarize_joke")
def summarize_joke_with_gemini(joke: str) -> str:
    # "joke-quality" is attached to ALL spans created in this function
    return call_llm(joke)
```

**Note**: The evaluator is attached to:
- Any spans created directly by the decorated function
- All child spans created within those spans
- Auto-instrumented LLM calls within the function

### With Sample Rate

Control what percentage of calls get evaluated:

```python
@evaluator(["hallucination-check"], sample_rate=0.5)
@observe_generation(name="my.llm.call")
def call_llm(prompt: str) -> str:
    # "hallucination-check" propagates to all child spans
    return sdk.generate(prompt)
```

### Multiple Evaluators

Attach multiple evaluators at once:

```python
@evaluator(
    slugs=["response-quality", "hallucination-check"],
    sample_rate=0.8,
    metadata=lambda query, results_count, **kwargs: {
        "query_length": len(query),
        "results_returned": results_count,
        "source": "vector_db",
    }
)
def search_knowledge_base(query: str, top_k: int = 5) -> list[dict]:
    results = [
        {"id": f"doc-{i}", "score": 0.9 - (i * 0.1), "content": f"Result {i}"}
        for i in range(min(top_k, 3))
    ]
    return results
```


## Using Context Managers

### attach_evaluator

The `attach_evaluator` context manager **both attaches to a specific span AND propagates to child spans**.

```python
from basalt.observability import trace_generation, attach_evaluator

with trace_generation("my.llm") as llm_span:
    with attach_evaluator("hallucination-check", span=llm_span):
        llm_span.set_model("gpt-4")
        llm_span.set_prompt("Hello")

        # Any child spans created here will also get "hallucination-check"
        result = call_llm()  # If this creates child spans, they inherit the evaluator
        llm_span.set_completion(result)
```

**Behavior**:
- Attaches "hallucination-check" to `llm_span`
- Also propagates "hallucination-check" to any child spans created within the context

### Attaching to Current Span

If you don't specify a span, evaluators attach to the current active span and propagate:

```python
from basalt.observability import trace_generation, attach_evaluator

with trace_generation("my.llm") as llm_span:
    with attach_evaluator("llm-ctx-eval"):
        # Attaches to current span (my.llm) AND propagates to children
        llm_span.set_prompt("Hello")
        llm_span.set_completion("Hi there!")
```


## Direct Span Methods

These methods attach evaluators **only to the specific span**, without propagating to children.

### add_evaluator()

Add a single evaluator with optional metadata to a specific span only:

```python
with trace_span("test.span") as span:
    span.add_evaluator(
        "test-eval",
        metadata={
            "source": "wikipedia",
            "confidence": 0.8,
        },
    )
    # Only this span gets "test-eval", not its children
```

### add_evaluators()

Add multiple evaluators at once to a specific span only:

```python
with trace_generation("test.llm") as span:
    span.add_evaluators("hallucination-check", "quality-eval")
    span.set_model("gpt-4")
    # Only this span gets the evaluators, not child spans
```

## Helper Functions

These helper functions also attach evaluators **only to the specified span**, without propagation.

### attach_evaluators_to_span()

Directly attach evaluators to a specific span handle only:

```python
from basalt.observability import trace_generation, attach_evaluators_to_span

with trace_generation("my.llm") as llm_span:
    attach_evaluators_to_span(llm_span, "hallucination-check", "quality-eval")
    # Only llm_span gets the evaluators, not its children
```

### attach_evaluators_to_current_span()

Attach evaluators to the currently active span only:

```python
from basalt.observability import attach_evaluators_to_current_span

# Inside an instrumented function or span context
attach_evaluators_to_current_span("hallucination-check", "quality-eval")
# Only the current span gets evaluators, not children
```

## Evaluator Configuration

### Setting Sample Rate

Control what percentage of spans get evaluated:

```python
from basalt.observability import trace_span, EvaluationConfig

with trace_span("test.span") as span:
    config = EvaluationConfig(sample_rate=0.5)  # 50% sampling
    span.set_evaluator_config(config)
    span.add_evaluator("test-eval")
```

Or use a dictionary:

```python
with trace_span("test.span") as span:
    span.set_evaluator_config({"sample_rate": 0.75})  # 75% sampling
    span.add_evaluator("test-eval")
```

### Setting Evaluator Metadata

Add metadata that applies to all evaluators on a span:

```python
with trace_span("test.span") as span:
    span.set_evaluator_metadata({
        "environment": "production",
        "user_segment": "premium"
    })
    span.add_evaluator("quality-eval")
```

## Integration with LLM Tracing

### Auto-Instrumented LLM Calls

Evaluators work seamlessly with auto-instrumented LLM providers:

```python
from basalt.observability import evaluator
from openai import OpenAI

@evaluator(
    slugs=["openai-quality", "toxicity"],
    sample_rate=1.0,
    metadata=lambda prompt, **kwargs: {
        "prompt_length": len(prompt),
        "context": "customer_support",
    }
)
def call_openai_with_evaluators(prompt: str) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Auto-instrumented by opentelemetry-instrumentation-openai
    # Evaluators automatically attach via BasaltCallEvaluatorProcessor
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
    )
    return response.choices[0].message.content
```


### Manual LLM Spans

```python
from basalt.observability import trace_generation

with trace_generation("test.llm") as span:
    span.set_model("gpt-4")
    span.add_evaluator("llm-eval")
    span.set_prompt("Explain quantum computing")
    result = model.generate("Explain quantum computing")
    span.set_completion(result)
```


## Setting Global Default Evaluators

### Using configure_trace_defaults()

Set evaluators that automatically apply to all new spans:

```python
from basalt.observability import configure_trace_defaults

configure_trace_defaults(
    evaluators=["default-evaluator", "safety-check"]
)
```

### Adding to Existing Defaults

Append evaluators without replacing existing defaults:

```python
from basalt.observability import add_default_evaluators

add_default_evaluators("new-evaluator")
```

## Advanced: with_evaluators() Context Manager

For advanced use cases, the `with_evaluators()` context manager provides low-level control over evaluator propagation:

```python
from basalt.observability.context_managers import with_evaluators
from basalt.observability import EvaluationConfig, trace_span

# Explicit propagation using with_evaluators
with with_evaluators(
    evaluators=["custom-eval"],
    config=EvaluationConfig(sample_rate=0.5),
    metadata={"source": "api"}
):
    # All spans created here get "custom-eval"
    with trace_span("parent"):  # Gets "custom-eval"
        with trace_span("child"):  # Also gets "custom-eval" (propagated)
            pass
```

**When to use**:
- You need explicit control over propagation scope
- You want to propagate evaluators without using the `@evaluator` decorator
- You're building custom instrumentation or middleware


### How Propagation Works

Propagation is implemented through:
1. **Context Storage**: `with_evaluators()` stores evaluators in OpenTelemetry context
2. **Automatic Application**: `BasaltCallEvaluatorProcessor` reads from context and applies to new spans
3. **Hierarchical Merging**: Child spans inherit parent evaluators plus their own


**Example - Understanding Propagation**:
```python
from basalt.observability import evaluator, trace_span

@evaluator("decorator-eval")
def parent_operation():
    with trace_span("parent.span") as parent:
        # Parent gets "decorator-eval" from decorator propagation
        parent.add_evaluator("parent-only")  # Add non-propagating evaluator

        child_operation()

def child_operation():
    with trace_span("child.span") as child:
        # Child gets "decorator-eval" (propagated from decorator)
        # Child does NOT get "parent-only" (direct attachment doesn't propagate)
        child.add_evaluator("child-only")  # Only on this span
```

**Result**:
- `parent.span`: `["decorator-eval", "parent-only"]`
- `child.span`: `["decorator-eval", "child-only"]`


## Span Attributes

Evaluators are stored on spans using these attributes:

- `basalt.span.evaluators` - Array of evaluator slugs
- `basalt.span.evaluators.config` - JSON-serialized EvaluationConfig
- `basalt.span.evaluator.metadata.*` - Individual metadata key-value pairs


## Complete Example

Here's a comprehensive example combining multiple evaluator patterns:

```python
from basalt import Basalt, TelemetryConfig
from basalt.observability import (
    trace_span,
    trace_generation,
    evaluator,
    configure_trace_defaults,
)

# Initialize Basalt with telemetry
telemetry = TelemetryConfig(
    service_name="evaluator-demo",
    environment="production",
    enable_llm_instrumentation=True,
)
basalt = Basalt(api_key="your-api-key", telemetry_config=telemetry)

# Set global default evaluators
configure_trace_defaults(evaluators=["safety-check"])

# Decorator-based evaluators
@evaluator(
    slugs=["response-quality", "hallucination-check"],
    sample_rate=0.8,
    metadata=lambda query, **kwargs: {
        "query_length": len(query),
        "source": "user_input",
    }
)
def search_and_generate(query: str) -> str:
    # Search knowledge base
    with trace_span("search.knowledge_base") as search_span:
        search_span.add_evaluator("retrieval-quality")
        results = search_database(query)

    # Generate response
    with trace_generation("llm.generate_answer") as gen_span:
        gen_span.set_model("gpt-4")
        gen_span.add_evaluator("answer-correctness")
        gen_span.set_prompt(f"Answer: {query}")

        response = call_llm(query, results)
        gen_span.set_completion(response)

    return response

# Run the function
result = search_and_generate("What is OpenTelemetry?")

# Shutdown to flush traces
basalt.shutdown()
```