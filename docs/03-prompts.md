# Prompts Management

Basalt's Prompts API provides version control, tag-based deployment, and dynamic variable substitution for your AI prompts. This guide covers all prompt operations with complete examples.

## Overview

The Prompts API enables you to:
- **Store prompts centrally** with version control
- **Deploy prompts** using tags (e.g., `production`, `staging`, `latest`)
- **Substitute variables** dynamically using Jinja2 templates
- **Cache prompts** for performance with graceful fallback
- **Manage model configurations** alongside prompts

## Listing Prompts

Retrieve all prompts accessible to your API key.

### Basic Listing

```python
from basalt import Basalt

basalt = Basalt(api_key="your-api-key")

# List all prompts
response = basalt.prompts.list_sync()

print(f"Total prompts: {response.total}")
print(f"Prompts returned: {len(response.prompts)}")

for prompt in response.prompts:
    print(f"\nSlug: {prompt.slug}")
    print(f"  Description: {prompt.description}")
    print(f"  Latest version: {prompt.latest_version}")
    print(f"  Tags: {', '.join(prompt.tags)}")

basalt.shutdown()
```

### Async Listing

```python
import asyncio
from basalt import Basalt

async def list_prompts_async():
    basalt = Basalt(api_key="your-api-key")

    response = await basalt.prompts.list_async()

    for prompt in response.prompts:
        print(f"{prompt.slug}: {prompt.description}")

    basalt.shutdown()

# Run async function
asyncio.run(list_prompts_async())
```

### Response Structure

The `PromptListResponse` contains:

```python
response.total         # Total number of prompts (int)
response.prompts       # List of prompt metadata
```

Each prompt in the list has:
- `slug` - Unique identifier
- `description` - Human-readable description
- `latest_version` - Most recent version number
- `tags` - List of tags (e.g., `['production', 'latest']`)

## Getting Prompts

Retrieve a specific prompt with its full content and configuration.

### Get Latest Version

```python
from basalt import Basalt

basalt = Basalt(api_key="your-api-key")

# Get the latest version
prompt = basalt.prompts.get_sync(
    slug='customer-greeting',
    tag='latest'
)

print(f"Prompt text: {prompt.text}")
print(f"Model: {prompt.model.provider}/{prompt.model.model}")
print(f"Version: {prompt.version}")

basalt.shutdown()
```

### Get Specific Version

```python
# Get a specific version by version number
prompt = basalt.prompts.get_sync(
    slug='customer-greeting',
    version='1.2.0'
)

print(f"Retrieved version: {prompt.version}")
```

### Get by Tag

```python
# Get by environment tag
production_prompt = basalt.prompts.get_sync(
    slug='customer-greeting',
    tag='production'
)

staging_prompt = basalt.prompts.get_sync(
    slug='customer-greeting',
    tag='staging'
)
```

### Prompt Object Structure

The returned `Prompt` object contains:

```python
prompt.slug                           # Prompt identifier
prompt.version                        # Version number (e.g., "1.2.0")
prompt.text                           # The actual prompt text
prompt.description                    # Human-readable description

# Model configuration
prompt.model.provider                 # "openai", "anthropic", etc.
prompt.model.model                    # "gpt-4", "claude-4-opus", etc.

# Model parameters (when available)
prompt.model.parameters.temperature   # Sampling temperature
prompt.model.parameters.max_tokens    # Maximum completion tokens
prompt.model.parameters.top_p         # Nucleus sampling parameter
```

### Accessing Model Parameters

```python
prompt = basalt.prompts.get_sync(slug='my-prompt', tag='latest')

# Use prompt configuration with your LLM client
import openai

client = openai.OpenAI()

response = client.chat.completions.create(
    model=prompt.model.model,
    messages=[{"role": "user", "content": prompt.text}],
    temperature=prompt.model.parameters.temperature or 0.7,
    max_tokens=prompt.model.parameters.max_tokens or 1000
)

print(response.choices[0].message.content)
```

## Variable Substitution

Prompts support Jinja2 template syntax for dynamic variable substitution.

### Basic Variable Substitution

```python
from basalt import Basalt

basalt = Basalt(api_key="your-api-key")

# Prompt in Basalt platform:
# "Hello {{customer_name}}, welcome to {{product_name}}!"

prompt = basalt.prompts.get_sync(
    slug='welcome-message',
    tag='latest',
    variables={
        'customer_name': 'Alice',
        'product_name': 'Premium Plan'
    }
)

print(prompt.text)
# Output: "Hello Alice, welcome to Premium Plan!"

basalt.shutdown()
```

## Observability with Context Managers

Prompts can be used as context managers to automatically inject prompt data to instrumented calls.
This enables generations spans to link back to the prompt used.

### Sync Context Manager Pattern

```python
from basalt import Basalt
import openai

basalt = Basalt(api_key="your-api-key")
client = openai.OpenAI()

# Use the prompt as a context manager for observability
with basalt.prompts.get_sync(
    slug='qa-prompt',
    tag='production',
    variables={"context": "Paris is the capital of France"}
) as prompt:
    # Any LLM calls here automatically nest under the prompt span
    response = client.chat.completions.create(
        model=prompt.model.model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt.text}
        ],
        temperature=prompt.model.parameters.temperature or 0.7,
        max_tokens=prompt.model.parameters.max_tokens or 1000
    )
    
    print(response.choices[0].message.content)

basalt.shutdown()
```

### Async Context Manager Pattern

```python
import asyncio
from basalt import Basalt
import openai

async def qa_workflow():
    basalt = Basalt(api_key="your-api-key")
    client = openai.AsyncOpenAI()
    
    # Async context manager for async LLM calls
    async with await basalt.prompts.get(
        slug='qa-prompt',
        tag='production',
        variables={"context": "Berlin is the capital of Germany"}
    ) as prompt:
        # Async LLM calls automatically nest under the prompt span
        response = await client.chat.completions.create(
            model=prompt.model.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt.text}
            ],
            temperature=prompt.model.parameters.temperature or 0.7,
            max_tokens=prompt.model.parameters.max_tokens or 1000
        )
        
        print(response.choices[0].message.content)
    
    basalt.shutdown()

# Run async workflow
asyncio.run(qa_workflow())
```


## Describing Prompts

Get metadata about a prompt including available versions and tags.

### Get Prompt Metadata

```python
from basalt import Basalt

basalt = Basalt(api_key="your-api-key")

# Describe a prompt
description = basalt.prompts.describe_sync(slug='customer-greeting')

print(f"Slug: {description.slug}")
print(f"Description: {description.description}")
print(f"Available versions: {description.available_versions}")
print(f"Available tags: {description.available_tags}")

basalt.shutdown()
```

### Complete Describe Example

```python
from basalt import Basalt

basalt = Basalt(api_key="your-api-key")

# Get prompt information
info = basalt.prompts.describe_sync('my-prompt')

print(f"\n=== Prompt: {info.slug} ===")
print(f"Description: {info.description}")

print(f"\nVersions ({len(info.available_versions)}):")
for version in info.available_versions:
    print(f"  - {version}")

print(f"\nTags ({len(info.available_tags)}):")
for tag in info.available_tags:
    print(f"  - {tag}")

# Now fetch a specific version based on the info
if 'production' in info.available_tags:
    prompt = basalt.prompts.get_sync(slug=info.slug, tag='production')
    print(f"\nProduction prompt: {prompt.text[:100]}...")

basalt.shutdown()
```

## Publishing Prompts

Assign or update tags for prompt versions.

### Publish to a Tag

```python
from basalt import Basalt

basalt = Basalt(api_key="your-api-key")

# Publish version 1.2.0 to the 'production' tag
result = basalt.prompt.publish_sync(
    slug='customer-greeting',
    new_tag='production',
    version='1.2.0'
)

print(f"Published: {result.slug} v{result.version} → tag '{result.new_tag}'")

basalt.shutdown()
```

### Async Publish

```python
import asyncio
from basalt import Basalt

async def publish_async():
    basalt = Basalt(api_key="your-api-key")
    result = await basalt.prompt.publish(
        slug="customer-greeting",
        new_tag="staging",
        version="1.2.0",
    )
    print(f"Published: {result.slug} v{result.version} → tag '{result.new_tag}'")
    basalt.shutdown()

asyncio.run(publish_async())
```

### Publishing Workflow Example

```python
from basalt import Basalt

def deploy_prompt_to_production(slug: str, version: str):
    """Deploy a tested prompt version to production"""
    basalt = Basalt(api_key="your-api-key")

    try:
        # First, verify the version exists
        info = basalt.prompts.describe_sync(slug)
        if version not in info.available_versions:
            print(f"Error: Version {version} not found!")
            return False

        # Publish to production
        result = basalt.prompt.publish_sync(
            slug=slug,
            new_tag='production',
            version=version
        )

        print(f"✓ Successfully deployed {slug} v{version} to production")
        return True

    finally:
        basalt.shutdown()

# Use the deployment function
deploy_prompt_to_production('customer-greeting', '2.1.0')
```

## Caching Behavior

Basalt implements a two-tier caching strategy for high availability:

### Cache Architecture

1. **Primary Cache**: 5-minute TTL for performance
2. **Fallback Cache**: Indefinite TTL for availability (used when API is unreachable)

### How Caching Works

```python
from basalt import Basalt

basalt = Basalt(api_key="your-api-key")

# First call: Fetches from API, stores in cache
prompt1 = basalt.prompts.get_sync(slug='my-prompt', tag='latest')
# Response time: ~200ms (network call)

# Second call within 5 minutes: Returns from cache
prompt2 = basalt.prompts.get_sync(slug='my-prompt', tag='latest')
# Response time: <1ms (cache hit)

# If API is down, returns from fallback cache
# (may be stale, but application continues working)
```

### Cache Keys

Caches are keyed by:
- Prompt slug
- Version (if specified)
- Tag (if specified)

```python
# These create different cache entries:
prompt_a = basalt.prompts.get_sync(slug='greeting', tag='latest')
prompt_b = basalt.prompts.get_sync(slug='greeting', tag='production')
prompt_c = basalt.prompts.get_sync(slug='greeting', version='1.0.0')
```

### Cache Behavior During Outages

```python
from basalt import Basalt
from basalt.types.exceptions import NetworkError

basalt = Basalt(api_key="your-api-key")

# First, populate the cache (when API is healthy)
prompt = basalt.prompts.get_sync(slug='my-prompt', tag='latest')

# Later, if API is down, fallback cache is used automatically
# No exception is raised; you get the last known good version
try:
    prompt = basalt.prompts.get_sync(slug='my-prompt', tag='latest')
    print("✓ Got prompt (possibly from fallback cache)")
except NetworkError:
    # This only happens if the prompt was never cached
    print("✗ Prompt unavailable and not in cache")
```

## Error Handling

Robust error handling for prompt operations.

### Common Exceptions

```python
from basalt import Basalt
from basalt.types.exceptions import (
    NotFoundError,
    UnauthorizedError,
    NetworkError,
    BasaltAPIError
)

basalt = Basalt(api_key="your-api-key")

try:
    prompt = basalt.prompts.get_sync(
        slug='my-prompt',
        tag='production'
    )
except NotFoundError:
    print("Error: Prompt not found or tag doesn't exist")
except UnauthorizedError:
    print("Error: Invalid API key")
except NetworkError as e:
    print(f"Error: Network issue - {e}")
except BasaltAPIError as e:
    print(f"Error: API error - {e}")

basalt.shutdown()
```

### Comprehensive Error Handling

```python
from basalt import Basalt
from basalt.types.exceptions import (
    NotFoundError,
    UnauthorizedError,
    NetworkError,
    BasaltAPIError
)

def get_prompt_with_fallback(slug: str, tag: str, fallback_text: str):
    """Get prompt with fallback to hardcoded text"""
    basalt = Basalt(api_key="your-api-key")

    try:
        prompt = basalt.prompts.get_sync(slug=slug, tag=tag)
        return prompt.text

    except NotFoundError:
        print(f"Warning: Prompt '{slug}' with tag '{tag}' not found")
        return fallback_text

    except NetworkError:
        print("Warning: Network error, using fallback")
        return fallback_text

    except BasaltAPIError as e:
        print(f"Warning: API error ({e}), using fallback")
        return fallback_text

    finally:
        basalt.shutdown()

# Usage
prompt_text = get_prompt_with_fallback(
    slug='greeting',
    tag='production',
    fallback_text="Hello! How can I help you today?"
)
```

## Observability Integration

Basalt prompts integrate seamlessly with observability through context managers. When you fetch a prompt, you can use it as a context manager to automatically create a span that captures prompt metadata and scopes any nested LLM calls.

### Context Manager Pattern

```python
from basalt import Basalt

basalt = Basalt(api_key="your-api-key")

# Sync context manager
with basalt.prompts.get_sync("qa-prompt", variables={"context": "Paris is capital"}) as prompt:
    # Creates a span that captures prompt metadata
    # Any auto-instrumented LLM calls here nest under this span
    response = openai.chat.completions.create(
        model=prompt.model.model,
        messages=[{"role": "user", "content": prompt.text}]
    )

# Async context manager
async with await basalt.prompts.get("qa-prompt", variables={"context": "Berlin is capital"}) as prompt:
    # Async LLM calls nest under this span
    response = await async_llm_client.generate(prompt.text)
```

### Imperative Pattern
```python
# Imperative usage creates an immediate span that ends right away
prompt = basalt.prompts.get_sync("qa-prompt", variables={"context": "London is capital"})

# LLM calls here create separate spans
response = openai.chat.completions.create(
    model=prompt.model.model,
    messages=[{"role": "user", "content": prompt.text}]
)
```

### When to Use Each Pattern

**Use context manager when:**
- You want to group auto-instrumented LLM calls under a prompt span
- You're using auto-instrumentation for OpenAI, Anthropic, etc.
- You want a clear trace hierarchy showing prompt → LLM call

**Use imperative when:**
- You need the prompt but don't need observability grouping
- You're manually instrumenting with `@observe` decorators

### Using with Manual Observability

Combine prompt context managers with manual spans:

```python
from basalt import Basalt
from basalt.observability import start_observe, observe

@start_observe(feature_slug="qa-system", name="QA System", identity={"user": {"id": "user_123"}})
def answer_question(question: str):
    with basalt.prompts.get_sync("qa-prompt", variables={"question": question}) as prompt:
        # Prompt span is a child of "QA System"
        response = openai.chat.completions.create(
            model=prompt.model.model,
            messages=[{"role": "user", "content": prompt.text}]
        )
        return response.choices[0].message.content
```

## Complete Examples

### Example 1: Dynamic Email Generator

```python
from basalt import Basalt
import openai

def generate_email(
    email_type: str,
    recipient_name: str,
    additional_context: dict
):
    """Generate personalized emails using Basalt prompts"""

    # Initialize clients
    basalt = Basalt(api_key="your-basalt-key")
    openai_client = openai.OpenAI(api_key="your-openai-key")

    try:
        # Get prompt with variables
        prompt = basalt.prompts.get_sync(
            slug='email-generator',
            tag='production',
            variables={
                'email_type': email_type,
                'recipient_name': recipient_name,
                **additional_context
            }
        )

        # Generate email
        response = openai_client.chat.completions.create(
            model=prompt.model.model,
            messages=[{"role": "user", "content": prompt.text}],
            temperature=prompt.model.parameters.temperature
        )

        return response.choices[0].message.content

    finally:
        basalt.shutdown()

# Usage
welcome_email = generate_email(
    email_type='welcome',
    recipient_name='Alice',
    additional_context={
        'product_name': 'Premium Plan',
        'trial_days': 14
    }
)

print(welcome_email)
```

### Example 2: Multi-Model Comparison

```python
from basalt import Basalt
import openai
import anthropic

def compare_models_for_prompt(slug: str, tag: str, user_input: str):
    """Compare different model responses for the same prompt"""

    basalt = Basalt(api_key="your-api-key")
    openai_client = openai.OpenAI()
    anthropic_client = anthropic.Anthropic()

    # Get prompt
    prompt = basalt.prompts.get_sync(
        slug=slug,
        tag=tag,
        variables={'user_input': user_input}
    )

    results = {}

    # Test with different models
    if prompt.model.provider == 'openai':
        response = openai_client.chat.completions.create(
            model=prompt.model.model,
            messages=[{"role": "user", "content": prompt.text}]
        )
        results['openai'] = response.choices[0].message.content

    if prompt.model.provider == 'anthropic':
        response = anthropic_client.messages.create(
            model=prompt.model.model,
            messages=[{"role": "user", "content": prompt.text}]
        )
        results['anthropic'] = response.content[0].text

    basalt.shutdown()
    return results

# Usage
responses = compare_models_for_prompt(
    slug='qa-assistant',
    tag='latest',
    user_input='Explain quantum computing'
)

for provider, response in responses.items():
    print(f"\n=== {provider.upper()} ===")
    print(response)
```

### Example 3: Prompt Version Testing

```python
from basalt import Basalt
from typing import List, Dict

def test_prompt_versions(
    slug: str,
    versions: List[str],
    test_cases: List[Dict[str, str]]
):
    """Test multiple prompt versions against test cases"""

    basalt = Basalt(api_key="your-api-key")

    results = {}

    for version in versions:
        print(f"\nTesting version {version}...")
        version_results = []

        for test_case in test_cases:
            try:
                prompt = basalt.prompts.get_sync(
                    slug=slug,
                    version=version,
                    variables=test_case
                )

                version_results.append({
                    'test_case': test_case,
                    'prompt_text': prompt.text,
                    'success': True
                })

            except Exception as e:
                version_results.append({
                    'test_case': test_case,
                    'error': str(e),
                    'success': False
                })

        results[version] = version_results

    basalt.shutdown()
    return results

# Usage
test_results = test_prompt_versions(
    slug='customer-support',
    versions=['1.0.0', '1.1.0', '2.0.0'],
    test_cases=[
        {'issue_type': 'billing', 'severity': 'high'},
        {'issue_type': 'technical', 'severity': 'low'},
        {'issue_type': 'account', 'severity': 'medium'}
    ]
)

# Analyze results
for version, results in test_results.items():
    success_rate = sum(1 for r in results if r['success']) / len(results)
    print(f"Version {version}: {success_rate:.0%} success rate")
```