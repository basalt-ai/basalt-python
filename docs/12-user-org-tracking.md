# User and Organization Tracking

Basalt allows you to associate traces with users and organizations, enabling you to track usage, costs, and performance metrics on a per-user or per-organization basis.

## Identity Structure

Identity information is provided as a dictionary with optional `user` and `organization` keys. Each key contains an entity with:
- `id` (required): Unique identifier
- `name` (optional): Display name

```python
from basalt.observability import Identity

identity: Identity = {
    "user": {
        "id": "user-123",
        "name": "John Doe"  # optional
    },
    "organization": {
        "id": "org-456",
        "name": "ACME Corp"  # optional
    }
}
```

## Setting Identity on Root Spans

### Using the Decorator

The most common approach is to set identity when creating a root span with `@start_observe`:

```python
from basalt.observability import start_observe

# Static identity
@start_observe(
    name="process_order",
    feature_slug="checkout",
    identity={
        "user": {"id": "user-123", "name": "John Doe"},
        "organization": {"id": "org-456", "name": "ACME Corp"}
    }
)
def process_order(order_id):
    # Identity automatically propagates to all child spans
    pass
```

### Using a Callable

You can also provide a callable that resolves identity from function arguments:

```python
def resolve_identity_from_request(request):
    """Extract identity from request context."""
    return {
        "user": {
            "id": request.user.id,
            "name": request.user.name
        },
        "organization": {
            "id": request.user.org_id,
            "name": request.user.org_name
        }
    }

@start_observe(
    name="api_endpoint",
    feature_slug="api",
    identity=resolve_identity_from_request
)
def handle_request(request):
    # Identity resolved from request argument
    pass
```

## Setting Identity Dynamically

You can also set identity dynamically within a span using `observe.set_identity()`:

```python
from basalt.observability import observe, start_observe

@start_observe(feature_slug="data-workflow", name="workflow")
def process_data(auth_token):
    # Authenticate and get user info
    user_info = authenticate(auth_token)
    
    # Set identity dynamically
    observe.set_identity({
        "user": {
            "id": user_info["id"],
            "name": user_info["name"]
        }
    })
    
    # Continue processing...
```

### On Span Handles

When using context managers, you can set identity on the span handle:

```python
from basalt.observability import async_start_observe

async def workflow():
    async with async_start_observe(feature_slug="task", name="process") as span:
        # Set identity on the span handle
        span.set_identity({
            "user": {"id": "user-789"},
            "organization": {"id": "org-101", "name": "Beta Corp"}
        })
        
        # Process...
```

## Identity Propagation

Identity set on a root span automatically propagates to all child spans. This means you only need to set it once at the entry point:

```python
@start_observe(
    name="order_pipeline",
    feature_slug="orders",
    identity={"user": {"id": "user-123"}}
)
def process_order():
    # Identity propagates to this child span
    @observe(kind=ObserveKind.SPAN, name="validate_order")
    def validate():
        pass
    
    # And to this child span
    @observe(kind=ObserveKind.SPAN, name="charge_payment")
    def charge():
        pass
    
    validate()
    charge()
```

## Partial Identity

You can set only user or only organization identity as needed:

```python
# User only
observe.set_identity({
    "user": {"id": "user-123", "name": "John Doe"}
})

# Organization only
observe.set_identity({
    "organization": {"id": "org-456"}
})

# Both (either can omit the name)
observe.set_identity({
    "user": {"id": "user-123"},
    "organization": {"id": "org-456", "name": "ACME Corp"}
})
```

## Best Practices

1. **Set identity early**: Set identity on the root span or as early as possible in the trace
2. **Use IDs consistently**: Ensure user and organization IDs are consistent across all traces
3. **Include names when available**: Names make traces easier to understand in the UI
4. **Leverage propagation**: Take advantage of automatic propagation to avoid redundant identity setting
5. **Handle authentication**: Set identity after authentication to ensure accurate tracking

## Example: Flask API

```python
from flask import Flask, request
from basalt.observability import start_observe, observe

app = Flask(__name__)

def get_identity_from_request():
    """Extract identity from Flask request context."""
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return None
    
    user = authenticate(auth_header)
    return {
        "user": {
            "id": user.id,
            "name": user.full_name
        },
        "organization": {
            "id": user.organization.id,
            "name": user.organization.name
        }
    }

@app.route("/api/orders/<order_id>")
@start_observe(feature_slug="orders", name="get_order", identity=get_identity_from_request)
def get_order(order_id):
    # Identity automatically set from request
    order = fetch_order(order_id)
    return {"order": order}
```

## Identity in Async Contexts

Identity works seamlessly with async/await:

```python
from basalt.observability import async_start_observe

@async_start_observe(
    feature_slug="background-jobs",
    name="async_workflow",
    identity={"user": {"id": "user-123"}}
)
async def process_async(data):
    # Identity propagates through async calls
    await step_one()
    await step_two()
```
