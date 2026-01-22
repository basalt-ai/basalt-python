"""Common type aliases used throughout the Basalt SDK."""

from typing import TypeAlias

# JSON-serializable types
# These represent data that can be safely serialized to/from JSON
JSONPrimitive: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONPrimitive | dict[str, "JSONValue"] | list["JSONValue"]
JSONDict: TypeAlias = dict[str, JSONValue]
JSONList: TypeAlias = list[JSONValue]

# OpenTelemetry span attributes can only be primitive types
SpanAttributeValue: TypeAlias = str | int | float | bool | None
