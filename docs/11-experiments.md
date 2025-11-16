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

### Attaching Experiments to Spans

The simplest way to track experiments is using `span.set_experiment()`:

```python
from basalt.observability import trace_span

with trace_span("experiment.variant_a") as span:
    span.set_experiment("exp-456", name="Model Comparison A/B Test")
    span.set_input({"variant": "A", "model": "gpt-4o"})

    result = run_test()

    span.set_output({"variant": "A", "result": result})
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
    name="GPT-4 vs GPT-3.5 Test"
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

## Running Experiments

### Basic A/B Test

Compare two variants of a model or prompt:

```python
from basalt.observability import trace_span

def run_ab_test():
    experiment_id = "exp-model-comparison"

    # Variant A: GPT-4o
    with trace_span("experiment.run_variant_a") as span:
        span.set_experiment(experiment_id, name="Model Comparison A/B Test")
        span.set_input({"variant": "A", "model": "gpt-4o"})

        result_a = call_model("gpt-4o", "Explain quantum computing")

        span.set_output({"variant": "A", "result": result_a})
        span.set_attribute("experiment.variant", "A")

    # Variant B: GPT-3.5-mini
    with trace_span("experiment.run_variant_b") as span:
        span.set_experiment(experiment_id, name="Model Comparison A/B Test")
        span.set_input({"variant": "B", "model": "gpt-3.5-mini"})

        result_b = call_model("gpt-3.5-mini", "Explain quantum computing")

        span.set_output({"variant": "B", "result": result_b})
        span.set_attribute("experiment.variant", "B")
```

### With Evaluators

Combine experiments with evaluators for automated quality assessment:

```python
from basalt.observability import trace_span

with trace_span("experiment.variant_a") as span:
    # Set experiment metadata
    span.set_experiment("exp-456", name="Model Comparison")

    # Attach evaluators to assess quality
    span.add_evaluator("consistency-check")
    span.add_evaluator("hallucination-check")

    # Run your variant
    result = run_variant_a()
```

### With LLM Tracing

Use `trace_generation` for LLM-specific experiments:

```python
from basalt.observability import trace_generation

with trace_generation("experiment.llm_variant_a") as span:
    span.set_experiment("exp-llm-temp", name="Temperature Comparison")
    span.set_model("gpt-4")
    span.set_prompt("Explain machine learning")

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Explain machine learning"}],
        temperature=0.7
    )

    span.set_completion(response.choices[0].message.content)
    span.set_attribute("experiment.temperature", 0.7)
```

## Setting Global Experiment Defaults

### Using Basalt Client

Set a default experiment for all traces:

```python
from basalt import Basalt
from basalt.observability import TraceExperiment

basalt = Basalt(
    api_key="your-api-key",
    trace_experiment=TraceExperiment(
        id="exp-123",
        name="Default Experiment",
        feature_slug="my-feature"
    )
)
```

Or using a dictionary:

```python
basalt = Basalt(
    api_key="your-api-key",
    trace_experiment={
        "id": "exp-123",
        "name": "Default Experiment",
        "feature_slug": "my-feature"
    }
)
```


### Using configure_trace_defaults()

Set experiment defaults globally:

```python
from basalt.observability import configure_trace_defaults, TraceExperiment

configure_trace_defaults(
    experiment=TraceExperiment(
        id="exp-global",
        name="Global Experiment",
        feature_slug="feature-x"
    )
)
```

## Experiments with Datasets

### Testing Prompts Against Datasets

Combine experiments with datasets for systematic testing:

```python
from basalt import Basalt
from basalt.observability import trace_span

basalt = Basalt(api_key="your-api-key")

def evaluate_prompt_against_dataset(prompt_slug: str, dataset_slug: str):
    """Test a prompt against a dataset"""

    # Get test dataset
    dataset = basalt.datasets.get_sync(slug=dataset_slug)

    results = []
    for row in dataset.rows:
        with trace_span(f"experiment.test_case.{row.name}") as span:
            span.set_experiment("exp-prompt-test", name="Prompt Testing")

            # Get prompt with row values as variables
            prompt = basalt.prompts.get_sync(
                slug=prompt_slug,
                tag='latest',
                variables=row.values
            )

            span.set_input({
                "test_case": row.name,
                "variables": row.values
            })

            # Execute test
            response = call_llm(prompt.rendered)

            # Compare with expected output
            match = response.strip() == row.ideal_output.strip()

            span.set_output({
                "expected": row.ideal_output,
                "actual": response,
                "match": match
            })

            results.append({
                'test_case': row.name,
                'match': match
            })

    return results
```

## Complete Experiment Workflow

### End-to-End Example

```python
from basalt import Basalt, TelemetryConfig
from basalt.observability import trace_span, trace_generation

# 1. Initialize Basalt with telemetry
telemetry = TelemetryConfig(
    service_name="experiment-demo",
    environment="production",
    enable_llm_instrumentation=True,
)
basalt = Basalt(api_key="your-api-key", telemetry_config=telemetry)

# 2. Create an experiment (optional - can use any ID)
from basalt.experiments import ExperimentsClient
exp_client = ExperimentsClient(api_key="your-api-key")
experiment = exp_client.create_sync(
    feature_slug="llm-routing",
    name="Model Comparison: GPT-4 vs Claude"
)

# 3. Run experiment with multiple variants
def run_model_comparison_experiment():
    """Compare different model variants."""

    experiment_id = experiment.id

    # Variant A: GPT-4
    with trace_span("experiment.variant_a") as span:
        span.set_experiment(
            experiment_id,
            name="Model Comparison: GPT-4 vs Claude",
            feature_slug="llm-routing"
        )
        span.add_evaluator("response-quality")
        span.add_evaluator("latency-check")

        span.set_input({
            "prompt": "Explain quantum computing",
            "model": "gpt-4",
            "temperature": 0.7
        })

        result_a = call_model("gpt-4", "Explain quantum computing")

        span.set_output({
            "model": "gpt-4",
            "result": result_a,
            "tokens_used": 150
        })
        span.set_attribute("experiment.variant", "A")

    # Variant B: Claude
    with trace_span("experiment.variant_b") as span:
        span.set_experiment(
            experiment_id,
            name="Model Comparison: GPT-4 vs Claude",
            feature_slug="llm-routing"
        )
        span.add_evaluator("response-quality")
        span.add_evaluator("latency-check")

        span.set_input({
            "prompt": "Explain quantum computing",
            "model": "claude-3-opus",
            "temperature": 0.7
        })

        result_b = call_model("claude-3-opus", "Explain quantum computing")

        span.set_output({
            "model": "claude-3-opus",
            "result": result_b,
            "tokens_used": 175
        })
        span.set_attribute("experiment.variant", "B")

# 4. Run the experiment
run_model_comparison_experiment()

# 5. Shutdown to flush traces
basalt.shutdown()
```

## Tracking Experiment Metadata

### Input and Output Tracking

Use `set_input()` and `set_output()` to track what goes into and comes out of each variant:

```python
with trace_span("experiment.variant") as span:
    span.set_experiment("exp-123", name="Test")

    # Track input parameters
    span.set_input({
        "variant": "A",
        "model": "gpt-4",
        "temperature": 0.7,
        "prompt": "Explain AI"
    })

    result = run_test()

    # Track output results
    span.set_output({
        "result": result,
        "tokens_used": 150,
        "latency_ms": 2000
    })
```


### Custom Attributes for Comparison

Add custom attributes for easier filtering and analysis:

```python
with trace_span("experiment.variant") as span:
    span.set_experiment("exp-123", name="Temperature Test")

    # Add custom attributes
    span.set_attribute("experiment.variant", "high_temp")
    span.set_attribute("experiment.model", "gpt-4")
    span.set_attribute("experiment.temperature", 0.9)
    span.set_attribute("experiment.iteration", 5)
```

### Variables Tracking

Track prompt variables used in experiments:

```python
with trace_generation("experiment.prompt_variant") as span:
    span.set_experiment("exp-prompts", name="Prompt Variations")

    variables = {
        "user_name": "Alice",
        "context": "customer support",
        "language": "English"
    }

    span.set_variables(variables)
    span.set_prompt(render_prompt(template, variables))
```

## Span Attributes

Experiment metadata is stored using these span attributes:

- `basalt.experiment.id` - Unique experiment identifier
- `basalt.experiment.name` - Human-readable experiment name
- `basalt.experiment.feature_slug` - Feature identifier for grouping


Example span with experiment attributes:

```json
{
    "name": "experiment.variant_a",
    "attributes": {
        "basalt.experiment.id": "exp-456",
        "basalt.experiment.name": "Model Comparison A/B Test",
        "basalt.experiment.feature_slug": "llm-routing",
        "experiment.variant": "A",
        "experiment.model": "gpt-4"
    }
}
```