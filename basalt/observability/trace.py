from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from opentelemetry.trace import Span, Tracer

from ..types.common import SpanAttributeValue
from . import semconv
from .context_managers import (
    get_current_otel_span,
    get_tracer,
)


class Trace:
    """
    Low-level tracing primitives for advanced users.
    Provides direct access to OpenTelemetry objects.
    """

    @staticmethod
    def current_span() -> Span | None:
        """Get the current OpenTelemetry span."""
        return get_current_otel_span()

    @staticmethod
    def get_tracer(name: str = "basalt.custom") -> Tracer:
        """Get an OpenTelemetry tracer."""
        return get_tracer(name)

    @staticmethod
    def add_event(name: str, attributes: Mapping[str, Any] | None = None) -> None:
        """Add a raw event to the current span."""
        span = get_current_otel_span()
        if span:
            span.add_event(name, attributes=attributes)

    @staticmethod
    def set_attribute(key: str, value: SpanAttributeValue) -> None:
        """Set a raw attribute on the current span."""
        span = get_current_otel_span()
        if span and value is not None:
            span.set_attribute(key, value)

    @staticmethod
    def set_attributes(attributes: Mapping[str, Any]) -> None:
        """Set multiple raw attributes on the current span."""
        span = get_current_otel_span()
        if span:
            span.set_attributes(attributes)

    @staticmethod
    def identify(
        user: str | dict[str, Any] | None = None,
        organization: str | dict[str, Any] | None = None,
    ) -> None:
        """
        Set or merge user and organization identity for the current span.

        Merges the provided identity with any existing identity attributes on the
        current span, overriding only the keys that are explicitly provided.

        Args:
            user: User identity as string ID or dict with 'id' and/or 'name' keys
            organization: Organization identity as string ID or dict with 'id' and/or 'name' keys

        Examples:
            # Set initial identity
            trace.identify(user={"id": "123", "name": "Alice"})

            # Update only name (ID preserved via merge)
            trace.identify(user={"name": "Alice Smith"})

            # Set organization separately
            trace.identify(organization="org-456")
        """
        span = get_current_otel_span()
        if not span:
            return

        current_user, current_org = _get_current_identity_from_span(span)

        if user is not None:
            _apply_identity_attributes(
                span=span,
                current=current_user,
                incoming=user,
                id_attr=semconv.BasaltUser.ID,
                name_attr=semconv.BasaltUser.NAME,
            )

        if organization is not None:
            _apply_identity_attributes(
                span=span,
                current=current_org,
                incoming=organization,
                id_attr=semconv.BasaltOrganization.ID,
                name_attr=semconv.BasaltOrganization.NAME,
            )


def _get_current_identity_from_span(span: Span) -> tuple[dict[str, str], dict[str, str]]:
    """Extract current user and org identity from span attributes."""
    user_dict = {}
    org_dict = {}

    attributes = getattr(span, "attributes", None)
    if isinstance(attributes, Mapping):
        if semconv.BasaltUser.ID in attributes:
            user_dict["id"] = str(attributes[semconv.BasaltUser.ID])
        if semconv.BasaltUser.NAME in attributes:
            user_dict["name"] = str(attributes[semconv.BasaltUser.NAME])
        if semconv.BasaltOrganization.ID in attributes:
            org_dict["id"] = str(attributes[semconv.BasaltOrganization.ID])
        if semconv.BasaltOrganization.NAME in attributes:
            org_dict["name"] = str(attributes[semconv.BasaltOrganization.NAME])

    return user_dict, org_dict


def _parse_identity_input(value: str | dict[str, Any] | None) -> dict[str, str]:
    """
    Parse identity input into normalized dict with 'id' and/or 'name' keys.

    Note: Empty strings and None values in dicts ARE included in the result,
    allowing explicit clearing/setting of attributes to empty values.
    """
    if value is None:
        return {}
    if isinstance(value, str):
        return {"id": value}
    if isinstance(value, dict):
        result = {}
        if "id" in value:
            result["id"] = str(value["id"]) if value["id"] is not None else ""
        if "name" in value:
            result["name"] = str(value["name"]) if value["name"] is not None else ""
        return result
    return {}


def _merge_identity(existing: dict[str, str], new: dict[str, str]) -> dict[str, str]:
    """Merge new identity values into existing, overriding keys present in new."""
    return {**existing, **new}


def _apply_identity_attributes(
    *,
    span: Span,
    current: dict[str, str],
    incoming: str | dict[str, Any] | None,
    id_attr: str,
    name_attr: str,
) -> None:
    new_identity = _parse_identity_input(incoming)
    merged_identity = _merge_identity(current, new_identity)
    if "id" in merged_identity:
        span.set_attribute(id_attr, merged_identity["id"])
    if "name" in merged_identity:
        span.set_attribute(name_attr, merged_identity["name"])


# Singleton instance
trace_api = Trace
