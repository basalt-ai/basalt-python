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
response = basalt.datasets.list_sync()

print(f"Total datasets: {response.total}")

for dataset in response.datasets:
    print(f"\nSlug: {dataset.slug}")
    print(f"  Name: {dataset.name}")
    print(f"  Description: {dataset.description}")
    print(f"  Rows: {dataset.num_rows}")
    print(f"  Columns: {len(dataset.columns)}")

basalt.shutdown()
```

### Async Listing

```python
import asyncio
from basalt import Basalt

async def list_datasets_async():
    basalt = Basalt(api_key="your-api-key")

    response = await basalt.datasets.list_async()

    for dataset in response.datasets:
        print(f"{dataset.slug}: {dataset.num_rows} rows")

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
dataset = basalt.datasets.get_sync(slug='customer-support-qa')

print(f"Dataset: {dataset.name}")
print(f"Description: {dataset.description}")
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

dataset = basalt.datasets.get_sync(slug='my-dataset')

# Iterate through columns
for column in dataset.columns:
    print(f"\nColumn: {column.name}")
    print(f"  Type: {column.type}")
    print(f"  Description: {column.description}")

basalt.shutdown()
```

## Dataset Structure

### Dataset Object

```python
dataset.slug           # Unique identifier
dataset.name           # Human-readable name
dataset.description    # Description
dataset.num_rows       # Number of rows
dataset.columns        # List of DatasetColumn objects
dataset.rows           # List of DatasetRow objects
```

### DatasetColumn Object

```python
column.name            # Column name
column.type            # Data type (e.g., "string", "number")
column.description     # Column description
```

### DatasetRow Object

```python
row.name               # Row identifier
row.values             # Dict[str, Any] - column_name -> value
row.ideal_output       # Optional[str] - Expected output for evaluation
row.metadata           # Optional[Dict[str, Any]] - Additional context
```

## Adding Rows

Add new test cases to existing datasets.

### Basic Row Addition

```python
from basalt import Basalt

basalt = Basalt(api_key="your-api-key")

# Add a row
row, warning = basalt.datasets.add_row_sync(
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
if warning:
    print(f"Warning: {warning}")

basalt.shutdown()
```

### Async Row Addition

```python
import asyncio
from basalt import Basalt

async def add_test_case():
    basalt = Basalt(api_key="your-api-key")

    row, warning = await basalt.datasets.add_row_async(
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

def add_multiple_test_cases(slug: str, test_cases: list):
    """Add multiple test cases to a dataset"""
    basalt = Basalt(api_key="your-api-key")

    added_rows = []

    for i, test_case in enumerate(test_cases):
        try:
            row, warning = basalt.datasets.add_row_sync(
                slug=slug,
                values=test_case['values'],
                name=test_case.get('name', f'test-{i:03d}'),
                ideal_output=test_case.get('ideal_output'),
                metadata=test_case.get('metadata', {})
            )

            added_rows.append(row)
            print(f"✓ Added: {row.name}")

            if warning:
                print(f"  Warning: {warning}")

        except Exception as e:
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
        row, _ = basalt.datasets.add_row_sync(
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
    dataset = basalt.datasets.get_sync(slug=dataset_slug)

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
        row, _ = basalt.datasets.add_row_sync(
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

## Complete Examples

### Example 1: Dataset-Driven Testing Framework

```python
from basalt import Basalt
from typing import Callable, Dict, Any, List
import openai

class DatasetTester:
    """Framework for testing AI functions against datasets"""

    def __init__(self, api_key: str):
        self.basalt = Basalt(api_key=api_key)
        self.openai = openai.OpenAI()

    def run_tests(
        self,
        dataset_slug: str,
        test_function: Callable,
        compare_function: Callable = None
    ) -> List[Dict[str, Any]]:
        """Run tests from a dataset"""

        # Get dataset
        dataset = self.basalt.datasets.get_sync(slug=dataset_slug)

        results = []

        for row in dataset.rows:
            print(f"\nTesting: {row.name}")

            try:
                # Run test function with row values
                actual_output = test_function(**row.values)

                # Compare results
                if compare_function:
                    match = compare_function(actual_output, row.ideal_output)
                else:
                    match = str(actual_output).strip() == str(row.ideal_output).strip()

                results.append({
                    'test_case': row.name,
                    'input': row.values,
                    'expected': row.ideal_output,
                    'actual': actual_output,
                    'passed': match,
                    'metadata': row.metadata
                })

                status = "✓ PASS" if match else "✗ FAIL"
                print(f"  {status}")

            except Exception as e:
                print(f"  ✗ ERROR: {e}")
                results.append({
                    'test_case': row.name,
                    'input': row.values,
                    'error': str(e),
                    'passed': False
                })

        return results

    def report(self, results: List[Dict[str, Any]]):
        """Generate test report"""
        total = len(results)
        passed = sum(1 for r in results if r.get('passed', False))
        failed = total - passed

        print(f"\n{'='*50}")
        print(f"TEST REPORT")
        print(f"{'='*50}")
        print(f"Total tests: {total}")
        print(f"Passed: {passed} ({passed/total:.1%})")
        print(f"Failed: {failed} ({failed/total:.1%})")

        if failed > 0:
            print(f"\nFailed tests:")
            for r in results:
                if not r.get('passed', False):
                    print(f"  - {r['test_case']}")
                    if 'error' in r:
                        print(f"    Error: {r['error']}")

    def shutdown(self):
        self.basalt.shutdown()

# Usage
tester = DatasetTester(api_key="your-api-key")

def my_qa_function(question: str, context: str) -> str:
    """Function to test"""
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"Context: {context}"},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content

# Run tests
results = tester.run_tests(
    dataset_slug='qa-test-suite',
    test_function=my_qa_function
)

# Generate report
tester.report(results)
tester.shutdown()
```

### Example 2: Continuous Dataset Improvement

```python
from basalt import Basalt
from datetime import datetime

class DatasetManager:
    """Manage and improve datasets over time"""

    def __init__(self, api_key: str):
        self.basalt = Basalt(api_key=api_key)

    def add_production_example(
        self,
        dataset_slug: str,
        input_data: dict,
        output: str,
        quality_score: float,
        source: str = "production"
    ):
        """Add high-quality production examples to dataset"""

        # Only add high-quality examples
        if quality_score < 0.8:
            print(f"Skipping low-quality example (score: {quality_score})")
            return None

        # Generate unique name
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"{source}-{timestamp}"

        # Add to dataset
        row, warning = self.basalt.datasets.add_row_sync(
            slug=dataset_slug,
            values=input_data,
            name=name,
            ideal_output=output,
            metadata={
                'quality_score': quality_score,
                'source': source,
                'timestamp': timestamp
            }
        )

        print(f"✓ Added {name} to dataset (quality: {quality_score:.2f})")

        return row

    def get_test_cases_by_category(
        self,
        dataset_slug: str,
        category: str
    ):
        """Filter test cases by category"""

        dataset = self.basalt.datasets.get_sync(slug=dataset_slug)

        filtered_rows = [
            row for row in dataset.rows
            if row.metadata and row.metadata.get('category') == category
        ]

        print(f"Found {len(filtered_rows)} rows in category '{category}'")
        return filtered_rows

    def shutdown(self):
        self.basalt.shutdown()

# Usage
manager = DatasetManager(api_key="your-api-key")

# Add production examples
manager.add_production_example(
    dataset_slug='support-qa',
    input_data={
        'question': 'How do I export my data?',
        'context': 'Data management'
    },
    output='Go to Settings > Data > Export. Choose format and click Download.',
    quality_score=0.95
)

# Get test cases by category
technical_cases = manager.get_test_cases_by_category(
    dataset_slug='support-qa',
    category='technical'
)

manager.shutdown()
```