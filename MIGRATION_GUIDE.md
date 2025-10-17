# Migration Guide: Tuple-based API to Exception-based API

## Overview

Starting with version 0.5.0, the Basalt Python SDK has introduced a new, cleaner API design. The old tuple-based error handling pattern `(error, result)` is deprecated and will be removed in **v1.0.0**.

This guide will help you migrate your code to the new exception-based API.

## Why This Change?

The new API design offers several improvements:

1. **Pythonic Error Handling**: Uses standard Python exceptions instead of tuple unpacking
2. **Cleaner Code**: No need to check error tuples in every call
3. **Better Type Safety**: IDEs and type checkers can provide better autocomplete and error detection
4. **Explicit Error Handling**: Errors are handled with try/except blocks, making error paths more explicit
5. **Consistent with Python Ecosystem**: Follows patterns used by popular libraries like `requests`, `httpx`, etc.

## What's Deprecated?

The following patterns are deprecated:

### Old Tuple-based API (Deprecated)
```python
from basalt import BasaltFacade

facade = BasaltFacade(api_key="your-api-key")

# Prompts - Returns (error, prompt, generation)
err, prompt, generation = facade.prompt.get_sync("greeting")
if err:
    print(f"Error: {err}")
    return
print(prompt.text)

# Datasets - Returns (error, datasets)
err, datasets = facade.datasets.list_sync()
if err:
    print(f"Error: {err}")
    return
```

## How to Migrate

### Step 1: Import New Clients

Instead of using `BasaltFacade`, import and use the new clients directly:

```python
from basalt.prompts import PromptsClient
from basalt.datasets import DatasetsClient
from basalt._internal.cache import Cache
from basalt.utils.logger import Logger
```

### Step 2: Initialize Clients

```python
# Setup dependencies
cache = Cache()
fallback_cache = Cache()  # Infinite cache for fallback
logger = Logger(log_level="warning")

# Initialize clients
prompts_client = PromptsClient(
    api_key="your-api-key",
    cache=cache,
    fallback_cache=fallback_cache,
    logger=logger
)

datasets_client = DatasetsClient(
    api_key="your-api-key",
    logger=logger
)
```

### Step 3: Update Error Handling

Replace tuple unpacking with try/except blocks:

#### Prompts Migration

**Before (Deprecated):**
```python
err, prompt, generation = facade.prompt.get_sync("greeting")
if err:
    print(f"Error: {err}")
    return

print(f"Prompt text: {prompt.text}")
print(f"Model: {prompt.model.model}")
```

**After (New API):**
```python
from basalt._internal.exceptions import NotFoundError, UnauthorizedError, BasaltAPIError

try:
    prompt, generation = prompts_client.get_sync("greeting")
    print(f"Prompt text: {prompt.text}")
    print(f"Model: {prompt.model.model}")
except NotFoundError as e:
    print(f"Prompt not found: {e.message}")
except UnauthorizedError as e:
    print(f"Authentication failed: {e.message}")
except BasaltAPIError as e:
    print(f"API error: {e.message}")
```

#### Datasets Migration

**Before (Deprecated):**
```python
err, datasets = facade.datasets.list_sync()
if err:
    print(f"Error: {err}")
    return

for dataset in datasets:
    print(f"Dataset: {dataset.name}")
```

**After (New API):**
```python
from basalt._internal.exceptions import BasaltAPIError

try:
    datasets = datasets_client.list_sync()
    for dataset in datasets:
        print(f"Dataset: {dataset.name}")
except BasaltAPIError as e:
    print(f"Error listing datasets: {e.message}")
```

#### Adding Dataset Rows

**Before (Deprecated):**
```python
err, row, warning = facade.datasets.add_row_sync(
    slug="my-dataset",
    values={"input": "Hello", "output": "Hi there"},
    name="example-1"
)
if err:
    print(f"Error: {err}")
    return
if warning:
    print(f"Warning: {warning}")
```

**After (New API):**
```python
try:
    row, warning = datasets_client.add_row_sync(
        slug="my-dataset",
        values={"input": "Hello", "output": "Hi there"},
        name="example-1"
    )
    if warning:
        print(f"Warning: {warning}")
    print(f"Added row: {row.name}")
except BasaltAPIError as e:
    print(f"Error adding row: {e.message}")
```

## Exception Hierarchy

The new API raises specific exceptions based on the error type:

```python
BasaltAPIError (base exception)
├── NetworkError           # Network/connection issues
├── UnauthorizedError      # 401 - Invalid API key
├── ForbiddenError         # 403 - Access denied
├── NotFoundError          # 404 - Resource not found
├── RateLimitError         # 429 - Rate limit exceeded
└── ServerError            # 500+ - Server errors
```

You can catch specific exceptions or use `BasaltAPIError` to catch all API errors:

```python
from basalt._internal.exceptions import (
    BasaltAPIError,
    NotFoundError,
    UnauthorizedError,
    RateLimitError
)

try:
    prompt, generation = prompts_client.get_sync("my-prompt")
except NotFoundError:
    print("Prompt doesn't exist")
except UnauthorizedError:
    print("Check your API key")
except RateLimitError:
    print("Too many requests, slow down")
except BasaltAPIError as e:
    print(f"Other API error: {e.message}")
```

## Async Support

Both old and new APIs support async operations:

**Before (Deprecated):**
```python
async def get_prompt():
    err, prompt, generation = await facade.prompt.get("greeting")
    if err:
        print(f"Error: {err}")
        return
    return prompt
```

**After (New API):**
```python
async def get_prompt():
    try:
        prompt, generation = await prompts_client.get("greeting")
        return prompt
    except BasaltAPIError as e:
        print(f"Error: {e.message}")
        return None
```

## Complete Example: Before & After

### Before (Deprecated)

```python
from basalt import BasaltFacade

facade = BasaltFacade(api_key="your-api-key")

# Get a prompt
err, prompt, generation = facade.prompt.get_sync(
    slug="greeting",
    variables={"name": "Alice"}
)
if err:
    print(f"Failed to get prompt: {err}")
    exit(1)

print(f"Prompt: {prompt.text}")

# List datasets
err, datasets = facade.datasets.list_sync()
if err:
    print(f"Failed to list datasets: {err}")
    exit(1)

for dataset in datasets:
    print(f"Dataset: {dataset.name}")
```

### After (New API)

```python
from basalt.prompts import PromptsClient
from basalt.datasets import DatasetsClient
from basalt._internal.cache import Cache
from basalt.utils.logger import Logger
from basalt._internal.exceptions import BasaltAPIError

# Setup
cache = Cache()
fallback_cache = Cache()
logger = Logger(log_level="warning")

prompts_client = PromptsClient(
    api_key="your-api-key",
    cache=cache,
    fallback_cache=fallback_cache,
    logger=logger
)

datasets_client = DatasetsClient(
    api_key="your-api-key",
    logger=logger
)

# Get a prompt
try:
    prompt, generation = prompts_client.get_sync(
        slug="greeting",
        variables={"name": "Alice"}
    )
    print(f"Prompt: {prompt.text}")
except BasaltAPIError as e:
    print(f"Failed to get prompt: {e.message}")
    exit(1)

# List datasets
try:
    datasets = datasets_client.list_sync()
    for dataset in datasets:
        print(f"Dataset: {dataset.name}")
except BasaltAPIError as e:
    print(f"Failed to list datasets: {e.message}")
    exit(1)
```

## Using BasaltFacade During Transition (Backward Compatibility)

If you're not ready to migrate immediately, you can continue using `BasaltFacade` with the old tuple-based API. However, you'll see deprecation warnings:

```python
from basalt import BasaltFacade
import warnings

# Suppress warnings if needed (not recommended for production)
warnings.filterwarnings("ignore", category=DeprecationWarning)

facade = BasaltFacade(api_key="your-api-key")
err, prompt, generation = facade.prompt.get_sync("greeting")
```

**Note**: This backward compatibility will be removed in v1.0.0. We strongly recommend migrating to the new API.

## Benefits Summary

### Old API Issues
- ❌ Requires error checking on every call
- ❌ Easy to forget to check errors
- ❌ Verbose tuple unpacking
- ❌ Harder to understand control flow
- ❌ Non-standard Python pattern

### New API Benefits
- ✅ Standard Python exception handling
- ✅ Clear error control flow with try/except
- ✅ Better IDE support and type checking
- ✅ More concise code
- ✅ Follows Python ecosystem conventions
- ✅ Easier to add custom error handling
- ✅ Better integration with logging and monitoring tools

## Migration Checklist

- [ ] Replace `BasaltFacade` imports with direct client imports
- [ ] Initialize `PromptsClient` and `DatasetsClient` directly
- [ ] Replace tuple unpacking `err, result = ...` with simple assignment
- [ ] Add try/except blocks around client calls
- [ ] Catch specific exceptions (`NotFoundError`, `UnauthorizedError`, etc.)
- [ ] Remove `if err:` error checks
- [ ] Update async functions similarly
- [ ] Test your changes thoroughly
- [ ] Remove any deprecation warning suppression

## Timeline

- **v0.5.0** (Current): New API introduced, old API deprecated with warnings
- **v1.0.0** (Planned): Old tuple-based API removed entirely

## Need Help?

- **Documentation**: https://docs.getbasalt.ai
- **GitHub Issues**: https://github.com/basalt-ai/basalt-python/issues
- **Discord Community**: https://discord.gg/basalt

## Frequently Asked Questions

### Q: Can I continue using the old API?

A: Yes, for now. The old API will continue to work until v1.0.0, but you'll see deprecation warnings. We recommend migrating as soon as possible.

### Q: Do I need to change my API key?

A: No, your API key remains the same. Only the SDK API design has changed.

### Q: Will the new API break my existing code?

A: If you continue using `BasaltFacade`, your code will continue to work (with warnings) until v1.0.0. The new clients are completely opt-in.

### Q: What about monitoring and generation tracking?

A: The new API still returns `Generation` objects for monitoring. Usage is the same:

```python
prompt, generation = prompts_client.get_sync("my-prompt")
# Use generation.log() or generation.trace() as before
```

### Q: Can I mix old and new APIs?

A: Technically yes, but we don't recommend it. Choose one approach and be consistent throughout your codebase.

### Q: How do I handle retries with the new API?

A: You can use standard Python retry libraries like `tenacity`:

```python
from tenacity import retry, stop_after_attempt, wait_exponential
from basalt._internal.exceptions import BasaltAPIError

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def get_prompt_with_retry(slug: str):
    return prompts_client.get_sync(slug)

try:
    prompt, generation = get_prompt_with_retry("my-prompt")
except BasaltAPIError as e:
    print(f"Failed after retries: {e.message}")
```

### Q: How do I handle fallback caching with errors?

A: The `PromptsClient` automatically handles fallback caching. If an API error occurs and a cached value exists in the fallback cache, it will be returned automatically:

```python
try:
    # This will use fallback cache if API fails
    prompt, generation = prompts_client.get_sync("my-prompt", cache_enabled=True)
except BasaltAPIError as e:
    # This will only raise if both API call fails AND no fallback cache exists
    print(f"No cached version available: {e.message}")
```

---

**Applies to**: basalt-python v0.5.0+
