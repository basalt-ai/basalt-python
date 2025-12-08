# Experiments

Experiments enable you to track A/B tests, model comparisons, and feature variations in your AI applications. They provide a structured way to compare different approaches and analyze their performance through observability traces.

## What are Experiments?

**Experiments** in Basalt are a mechanism for:
- Tracking different variants of prompts, models, or approaches
- Comparing performance across versions
- Associating traces with specific experiments
- Attaching experiment metadata to observability spans
- Running systematic tests and comparisons

Each experiment has:
- **ID**: Unique identifier (e.g., `"exp-456"`)
- **Name**: Human-readable description (e.g., `"Model Comparison A/B Test"`)
- **Feature Slug**: Optional feature identifier for grouping experiments

## Core Concepts

### Experiment Model

The `Experiment` class represents an experiment in the Basalt system:

```python
@dataclass(slots=True, frozen=True)
class Experiment:
    id: str                 # Unique identifier
    name: str               # Human-readable name
    feature_slug: str       # Feature slug
    created_at: str         # ISO 8601 timestamp
```

### TraceExperiment

The `TraceExperiment` class represents experiment metadata attached to spans:

```python
@dataclass(frozen=True, slots=True)
class TraceExperiment:
    id: str                      # Experiment ID
    name: str | None = None      # Optional display name
    feature_slug: str | None = None  # Optional feature slug
```

## Quick Start

### Attaching Experiments at Root Span

The recommended way to track experiments is using the `experiment` parameter on `start_observe`:

```python
from basalt.observability import start_observe, observe


@start_observe(
    feature_slug="ab-test",
    name="experiment.variant_a",
    experiment="exp-456",
    identity={
        "organization": {"id": "123", "name": "ACME"},
        "user": {"id": "456", "name": "John Doe"}
    }
)
def run_variant_a():
    # Experiment metadata automatically attached to root span
    observe.set_input({"variant": "A", "model": "gpt-4o"})

    result = run_test()

    observe.set_output({"variant": "A", "result": result})
    return result


# Or using context manager
with start_observe(
        feature_slug="ab-test",
        name="experiment.variant_b",
        experiment="exp-456",
):
    # Your experiment code
    pass
```



## Creating Experiments

### Using ExperimentsClient

Create experiments programmatically using the API client:

```python
from basalt.experiments import ExperimentsClient

client = ExperimentsClient(api_key="your-api-key")

# Create experiment synchronously
experiment = client.create_sync(
    feature_slug="my-feature",
    name="GPT-5.1 vs GPT-5 Test"
)

print(f"Created experiment: {experiment.id}")
# Output: Created experiment: exp_abc123
```

### Async Creation

```python
import asyncio
from basalt.experiments import ExperimentsClient

async def create_experiment():
    client = ExperimentsClient(api_key="your-api-key")

    experiment = await client.create(
        feature_slug="llm-routing",
        name="Model Selection Experiment"
    )

    return experiment

experiment = asyncio.run(create_experiment())
```
