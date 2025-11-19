# Advanced Tracing

For power users and complex integration scenarios, Basalt exposes the underlying OpenTelemetry primitives and allows for advanced configuration.

## Low-Level `trace` API

While the `observe` API covers most use cases, you may sometimes need direct access to the OpenTelemetry span or tracer. The `basalt.trace` module provides this low-level access.

```python
from basalt.observability import trace

# Get the current active OpenTelemetry span
span = trace.current_span()
if span:
    span.set_attribute("custom.low_level.attr", 123)

# Get the underlying OpenTelemetry tracer
tracer = trace.get_tracer("my.custom.tracer")
with tracer.start_as_current_span("manual_span") as span:
    span.add_event("something_happened")
```

### Available Methods

- `trace.current_span()`: Returns the current `opentelemetry.trace.Span` or `None`.
- `trace.get_tracer(name)`: Returns an `opentelemetry.trace.Tracer`.
- `trace.add_event(name, attributes)`: Adds an event to the current span.
- `trace.set_attribute(key, value)`: Sets a single attribute on the current span.
- `trace.set_attributes(dict)`: Sets multiple attributes.

## OpenTelemetry Configuration

Basalt is built on top of OpenTelemetry (OTEL). You can configure the underlying OTEL components using standard environment variables or Basalt's configuration helpers.

### Exporters

By default, Basalt exports traces to the Basalt platform. You can configure additional exporters (e.g., Console, OTLP) for debugging or dual-homing data.

#### Console Exporter (Debugging)

To print traces to the console (stdout), set:

```bash
export BASALT_TRACE_CONSOLE_EXPORTER="true"
```

#### Custom OTLP Exporter

To send traces to another OTLP endpoint (e.g., Jaeger, Honeycomb):

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
export OTEL_EXPORTER_OTLP_HEADERS="api-key=..."
```

### Sampling

Control the sampling rate of your traces.

```bash
# Sample 100% of traces (default)
export BASALT_TRACE_SAMPLE_RATE=1.0

# Sample 10% of traces
export BASALT_TRACE_SAMPLE_RATE=0.1
```

## Custom Providers

If you need to manually configure the `TracerProvider`, you can do so before initializing Basalt.

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

provider = TracerProvider()
processor = BatchSpanProcessor(ConsoleSpanExporter())
provider.add_span_processor(processor)

# Set the global provider
trace.set_tracer_provider(provider)
```

Note: If you set a custom provider, Basalt will use it, but you are responsible for ensuring it is correctly configured to export data to Basalt if desired.
