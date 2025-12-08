# Async Observability

Basalt provides full support for async observability with the same API semantics as the synchronous version. This guide covers async-specific patterns for tracing asynchronous workflows.

## Overview

Async observability in Basalt works identically to synchronous observability:
- `async_start_observe` creates root spans in async functions
- `async_observe` creates nested spans in async functions
- `@observe` decorator automatically detects async functions

## Quick Start

### Basic Async Tracing

```python
import asyncio
from basalt.observability import async_start_observe, async_observe

async def main():
    async with async_start_observe(feature_slug="async-workflow", name="async_workflow") as span:
        span.set_input({"task": "process_data"})
        result = await process_data()
        span.set_output(result)

async def process_data():
    async with async_observe(name="data_processing") as span:
        # Your async logic here
        await asyncio.sleep(0.1)
        return {"status": "complete"}

asyncio.run(main())
```

## Async Root Spans with async_start_observe

Use `async_start_observe` as the entry point for async traces:

```python
from basalt.observability import async_start_observe

# Context manager form
async def workflow():
    async with async_start_observe(
        feature_slug="user-request",
        name="User Request",
        identity={"user": {"id": "user_123"}},
        metadata={"environment": "production"}
    ) as span:
        span.set_input({"request_id": "req_123"})
        result = await process()
        span.set_output(result)

# Decorator form
@async_start_observe(
    feature_slug="api-handler",
    name="API Handler",
    identity={"user": {"id": "user_456"}}
)
async def handle_request(user_id: str):
    return await fetch_data(user_id)
```

## Async Nested Spans with async_observe

Create nested spans for async operations:

```python
from basalt.observability import async_observe, ObserveKind

# Context manager form
async def fetch_data(item_id: int):
    async with async_observe(name="fetch_data", kind=ObserveKind.RETRIEVAL) as span:
        span.set_input({"item_id": item_id})
        data = await database.fetch(item_id)
        span.set_output(data)
        return data

# Decorator form
@async_observe(name="process_item", kind=ObserveKind.FUNCTION)
async def process_item(item: dict) -> dict:
    await asyncio.sleep(0.1)
    return {"processed": True, "item": item}

# Generation spans for LLM calls
@async_observe(kind=ObserveKind.GENERATION)
async def generate_response(prompt: str):
    # Your async LLM call
    return await llm_client.generate(prompt)
```

## Automatic Async Detection

The `@observe` and `@start_observe` decorators automatically detect async functions:

```python
from basalt.observability import observe, start_observe

# These automatically work with async functions
@start_observe(feature_slug="async-root", name="Async Root")
async def async_root():
    await nested_operation()

@observe(name="Nested Async")
async def nested_operation():
    await asyncio.sleep(0.1)
```

No need to use `async_observe` or `async_start_observe` when using decorators - the regular decorators handle both sync and async.

## Complete Example

```python
import asyncio
from basalt.observability import async_start_observe, async_observe, ObserveKind

async def fetch_user(user_id: str) -> dict:
    """Fetch user data from database"""
    async with async_observe(name="fetch_user", kind=ObserveKind.RETRIEVAL) as span:
        span.set_input({"user_id": user_id})
        await asyncio.sleep(0.05)  # Simulate DB query
        user = {"id": user_id, "name": "Alice"}
        span.set_output(user)
        return user

async def generate_greeting(user: dict) -> str:
    """Generate personalized greeting using LLM"""
    async with async_observe(name="generate_greeting", kind=ObserveKind.GENERATION) as span:
        span.set_input(user)
        await asyncio.sleep(0.1)  # Simulate LLM call
        greeting = f"Hello {user['name']}!"
        span.set_output({"greeting": greeting})
        return greeting

async def handle_user_request(user_id: str) -> str:
    """Main handler with root span"""
    async with async_start_observe(
        feature_slug="user-request",
        name="user_request",
        identity={"user": {"id": user_id}},
        metadata={"version": "2.0"}
    ) as root_span:
        root_span.set_input({"user_id": user_id})

        # Fetch user data
        user = await fetch_user(user_id)

        # Generate greeting
        greeting = await generate_greeting(user)

        root_span.set_output({"greeting": greeting})
        return greeting

# Run the async workflow
asyncio.run(handle_user_request("user_123"))
```