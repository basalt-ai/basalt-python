"""
Utilities for converting OpenTelemetry spans to OTLP JSON format.

This module provides helpers for converting Python OpenTelemetry ReadableSpan objects
into the OTLP JSON structure that matches the TypeScript OtelJsonParser contract.

The OTLP JSON structure follows the OpenTelemetry Protocol specification:
https://github.com/open-telemetry/opentelemetry-proto/blob/main/opentelemetry/proto/trace/v1/trace.proto
"""

from __future__ import annotations

from typing import Any


def _to_otel_attribute_value(value: object) -> dict[str, object]:
    """
    Convert a Python attribute value to the OTLP JSON AttributeValue shape.

    The OTLP AttributeValue is a union type that can represent:
    - stringValue: string
    - boolValue: bool
    - intValue: int (as number or string in JSON)
    - doubleValue: float
    - arrayValue: { values: AttributeValue[] }
    - kvlistValue: { values: [{ key, value }] }
    - bytesValue: base64-encoded string

    Args:
        value: The Python value to convert (str, int, float, bool, list, dict)

    Returns:
        A dict with exactly one of the *Value keys set
    """
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": value}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, list):
        # For arrays, wrap each element as an AttributeValue
        return {
            "arrayValue": {
                "values": [_to_otel_attribute_value(v) for v in value],
            }
        }
    if isinstance(value, dict):
        # For dicts, treat them as kvlistValue
        return {
            "kvlistValue": {
                "values": [
                    {"key": k, "value": _to_otel_attribute_value(v)} for k, v in value.items()
                ]
            }
        }

    # Fallback: stringify
    return {"stringValue": str(value)}


def spans_to_otel_json(
    spans: list[Any],
    resource_attributes: dict[str, Any],
    scope_name: str,
    scope_version: str | None = None,
) -> dict[str, Any]:
    """
    Convert OpenTelemetry ReadableSpan objects into OTLP JSON structure.

    This function produces JSON that matches the OTLP trace format expected by
    TypeScript parsers and OTLP receivers:

    ```json
    {
      "resourceSpans": [
        {
          "resource": {
            "attributes": [{ "key": ..., "value": AttributeValue }, ...]
          },
          "scopeSpans": [
            {
              "scope": { "name": ..., "version": ... },
              "spans": [
                {
                  "traceId": "32-char hex",
                  "spanId": "16-char hex",
                  "parentSpanId": "16-char hex" | null,
                  "name": "span name",
                  "kind": 1-5,
                  "startTimeUnixNano": "nanoseconds",
                  "endTimeUnixNano": "nanoseconds",
                  "attributes": [{ "key": ..., "value": ... }],
                  "status": { "code": 0-2, "message": ... },
                  "events": [],
                  "links": []
                }
              ]
            }
          ]
        }
      ]
    }
    ```

    Args:
        spans: List of OpenTelemetry ReadableSpan objects to convert
        resource_attributes: Resource-level attributes (service.name, etc.)
        scope_name: Instrumentation scope name (e.g., "opentelemetry.instrumentation.openai")
        scope_version: Optional instrumentation scope version

    Returns:
        A dict representing the OTLP JSON structure
    """
    # Convert resource attributes to OTLP format
    resource_attrs_list = [
        {"key": k, "value": _to_otel_attribute_value(v)} for k, v in resource_attributes.items()
    ]

    otlp_spans = []
    for span in spans:
        # Format trace_id and span_id as hex strings (32 and 16 chars respectively)
        trace_id_hex = f"{span.context.trace_id:032x}"
        span_id_hex = f"{span.context.span_id:016x}"

        # Convert span attributes to OTLP format
        attributes_list = [
            {"key": k, "value": _to_otel_attribute_value(v)} for k, v in span.attributes.items()
        ]

        # Build OTLP span object
        otlp_span: dict[str, Any] = {
            "traceId": trace_id_hex,
            "spanId": span_id_hex,
            "parentSpanId": f"{span.parent.span_id:016x}" if span.parent is not None else None,
            "name": span.name,
            "kind": span.kind.value,  # SpanKind enum value (1-5)
            "startTimeUnixNano": str(span.start_time),
            "endTimeUnixNano": str(span.end_time),
            "attributes": attributes_list,
            "status": {
                "code": span._status.status_code.value,
                "message": span._status.description or None,
            },
            # For simplicity, keeping these empty; extend if needed
            "events": [],
            "links": [],
            "droppedAttributesCount": 0,
            "droppedEventsCount": 0,
            "droppedLinksCount": 0,
        }

        otlp_spans.append(otlp_span)

    # Build scope object
    scope_obj: dict[str, Any] = {"name": scope_name}
    if scope_version is not None:
        scope_obj["version"] = scope_version

    # Build complete OTLP structure
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": resource_attrs_list,
                },
                "scopeSpans": [
                    {
                        "scope": scope_obj,
                        "spans": otlp_spans,
                    }
                ],
            }
        ]
    }
