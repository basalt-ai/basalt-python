"""Legacy trace flusher removed in the OTEL rewrite."""

raise ImportError(
    "basalt.utils.flusher was removed. Use basalt.observability flush helpers"
    " instead."
)

