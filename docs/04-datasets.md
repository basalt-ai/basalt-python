# Datasets Management

Basalt's Datasets API helps you organize test data for evaluating AI model outputs. Store inputs, expected outputs, and metadata to systematically test and improve your AI applications.

## Overview

Datasets enable you to:
- **Organize test cases** with structured inputs and expected outputs
- **Store ideal outputs** for automated quality evaluation
- **Track metadata** for additional context on each test case
- **Integrate with evaluators** for quality monitoring

## Table of Contents

- [Listing Datasets](#listing-datasets)
- [Getting Datasets](#getting-datasets)
- [Adding Rows](#adding-rows)
- [Dataset Structure](#dataset-structure)
- [Use Cases](#use-cases)
- [Complete Examples](#complete-examples)

## Listing Datasets

Retrieve all datasets accessible to your API key.

### Basic Listing

```python
from basalt import Basalt

basalt = Basalt(api_key="your-api-key")

# List all datasets
datasets = basalt.datasets.list_sync()

print(f"Total datasets: {len(datasets)}")

for dataset in datasets:
    print(f"\nSlug: {dataset.slug}")
    print(f"  Name: {dataset.name}")
    print(f"  Rows: {len(dataset.rows)}")
    print(f"  Columns: {len(dataset.columns)}")

basalt.shutdown()
```

### Async Listing

```python
import asyncio
from basalt import Basalt

async def list_datasets_async():
    basalt = Basalt(api_key="your-api-key")

    datasets = await basalt.datasets.list()

    for dataset in datasets:
        print(f"{dataset.slug}: {len(dataset.rows)} rows")

    basalt.shutdown()

asyncio.run(list_datasets_async())
```

## Getting Datasets

Retrieve a specific dataset with all its rows and columns.

### Basic Get

```python
from basalt import Basalt

basalt = Basalt(api_key="your-api-key")

# Get dataset by slug
dataset = basalt.datasets.get_sync('customer-support-qa')

print(f"Dataset: {dataset.name}")
print(f"Slug: {dataset.slug}")
print(f"Total rows: {len(dataset.rows)}")
print(f"Columns: {[col.name for col in dataset.columns]}")

# Access rows
for row in dataset.rows:
    print(f"\nRow: {row.name}")
    print(f"  Values: {row.values}")
    print(f"  Ideal output: {row.ideal_output}")
    print(f"  Metadata: {row.metadata}")

basalt.shutdown()
```

### Accessing Column Information

```python
from basalt import Basalt

basalt = Basalt(api_key="your-api-key")

dataset = basalt.datasets.get_sync('my-dataset')

# Iterate through columns
for column in dataset.columns:
    print(f"\nColumn: {column.name}")
    print(f"  Type: {column.type}")

basalt.shutdown()
```

## Dataset Structure

### Dataset Object

```python
dataset.slug           # str - Unique identifier
dataset.name           # str - Human-readable name
dataset.columns        # list[DatasetColumn] - List of DatasetColumn objects
dataset.rows           # list[DatasetRow] - List of DatasetRow objects
```

### DatasetColumn Object

```python
column.name            # str - Column name
column.type            # str | None - Data type (optional)
```

### DatasetRow Object

```python
row.name               # str | None - Row identifier (optional)
row.values             # dict[str, str] - column_name -> value
row.ideal_output       # str | None - Expected output for evaluation (optional)
row.metadata           # dict[str, Any] - Additional context (defaults to empty dict)
```

## Adding Rows

Add new test cases to existing datasets.

### Basic Row Addition

```python
from basalt import Basalt

basalt = Basalt(api_key="your-api-key")

# Add a row
row = basalt.datasets.add_row_sync(
    slug='customer-support-qa',
    values={
        'question': 'How do I reset my password?',
        'context': 'User account management'
    },
    name='test-case-001',
    ideal_output='Click on "Forgot Password" on the login page.',
    metadata={'category': 'account', 'priority': 'high'}
)

print(f"Added row: {row.name}")
print(f"Row values: {row.values}")

basalt.shutdown()
```

### Async Row Addition

```python
import asyncio
from basalt import Basalt

async def add_test_case():
    basalt = Basalt(api_key="your-api-key")

    row = await basalt.datasets.add_row(
        slug='qa-dataset',
        values={'input': 'test input', 'category': 'support'},
        name='async-test-001',
        ideal_output='Expected response here'
    )

    print(f"Added: {row.name}")
    basalt.shutdown()

asyncio.run(add_test_case())
```

### Bulk Row Addition

```python
from basalt import Basalt
from basalt.types.exceptions import BasaltAPIError

def add_multiple_test_cases(slug: str, test_cases: list):
    """Add multiple test cases to a dataset"""
    basalt = Basalt(api_key="your-api-key")

    added_rows = []

    for i, test_case in enumerate(test_cases):
        try:
            row = basalt.datasets.add_row_sync(
                slug=slug,
                values=test_case['values'],
                name=test_case.get('name', f'test-{i:03d}'),
                ideal_output=test_case.get('ideal_output'),
                metadata=test_case.get('metadata', {})
            )

            added_rows.append(row)
            print(f"✓ Added: {row.name}")

        except BasaltAPIError as e:
            print(f"✗ Failed to add test case {i}: {e}")

    basalt.shutdown()
    return added_rows

# Usage
test_cases = [
    {
        'name': 'billing-001',
        'values': {
            'question': 'How do I update my billing info?',
            'category': 'billing'
        },
        'ideal_output': 'Go to Settings > Billing > Update Payment Method',
        'metadata': {'difficulty': 'easy'}
    },
    {
        'name': 'technical-001',
        'values': {
            'question': 'API rate limit exceeded',
            'category': 'technical'
        },
        'ideal_output': 'Your rate limit is 100 requests/min. Implement exponential backoff.',
        'metadata': {'difficulty': 'medium'}
    }
]

add_multiple_test_cases('support-qa', test_cases)
```

## Use Cases

### Use Case 1: RAG Evaluation Dataset

```python
from basalt import Basalt

def create_rag_test_dataset(slug: str):
    """Create a dataset for evaluating RAG systems"""
    basalt = Basalt(api_key="your-api-key")

    test_cases = [
        {
            'name': 'factual-001',
            'values': {
                'query': 'What is the capital of France?',
                'context': 'Geography question',
                'expected_retrieval': 'Paris'
            },
            'ideal_output': 'The capital of France is Paris.',
            'metadata': {
                'category': 'factual',
                'difficulty': 'easy',
                'requires_retrieval': False
            }
        },
        {
            'name': 'technical-001',
            'values': {
                'query': 'How do I authenticate with the API?',
                'context': 'API documentation query',
                'expected_retrieval': 'Authentication section'
            },
            'ideal_output': 'To authenticate, include your API key in the Authorization header.',
            'metadata': {
                'category': 'technical',
                'difficulty': 'medium',
                'requires_retrieval': True
            }
        }
    ]

    for test_case in test_cases:
        row = basalt.datasets.add_row_sync(
            slug=slug,
            **test_case
        )
        print(f"Added: {row.name}")

    basalt.shutdown()

create_rag_test_dataset('rag-evaluation')
```

### Use Case 2: Prompt Testing Dataset

```python
from basalt import Basalt

def evaluate_prompt_against_dataset(prompt_slug: str, dataset_slug: str):
    """Test a prompt against a dataset"""
    import openai

    basalt = Basalt(api_key="your-api-key")
    openai_client = openai.OpenAI()

    # Get dataset
    dataset = basalt.datasets.get_sync(dataset_slug)

    results = []

    for row in dataset.rows:
        # Get prompt with row values as variables
        prompt = basalt.prompts.get_sync(
            slug=prompt_slug,
            tag='latest',
            variables=row.values
        )

        # Generate response
        response = openai_client.chat.completions.create(
            model=prompt.model.model,
            messages=[{"role": "user", "content": prompt.text}]
        )

        actual_output = response.choices[0].message.content

        # Compare with ideal output
        results.append({
            'test_case': row.name,
            'input': row.values,
            'expected': row.ideal_output,
            'actual': actual_output,
            'match': actual_output.strip() == row.ideal_output.strip()
        })

    basalt.shutdown()

    # Report results
    total = len(results)
    passed = sum(1 for r in results if r['match'])

    print(f"\n=== Evaluation Results ===")
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success rate: {passed/total:.1%}")

    return results

# Usage
results = evaluate_prompt_against_dataset(
    prompt_slug='qa-assistant',
    dataset_slug='qa-test-cases'
)
```

### Use Case 3: Collecting Production Data for Evaluation

```python
from basalt import Basalt
from basalt.observability import observe_generation

basalt = Basalt(api_key="your-api-key", enable_telemetry=True)

@observe_generation(name="customer_support.generate")
def generate_support_response(question: str, context: str):
    """Generate customer support response"""
    import openai

    client = openai.OpenAI()

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful support agent."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ]
    )

    return response.choices[0].message.content

def save_to_dataset(question: str, context: str, response: str, human_rating: int):
    """Save good examples to dataset for future evaluation"""

    # Only save highly-rated responses
    if human_rating >= 4:
        row = basalt.datasets.add_row_sync(
            slug='support-golden-dataset',
            values={
                'question': question,
                'context': context
            },
            ideal_output=response,
            metadata={
                'human_rating': human_rating,
                'source': 'production'
            }
        )
        print(f"Saved to dataset: {row.name}")

# Usage in production
question = "How do I upgrade my plan?"
context = "Billing and subscriptions"

response = generate_support_response(question, context)
human_rating = 5  # From human review

save_to_dataset(question, context, response, human_rating)

basalt.shutdown()
```
