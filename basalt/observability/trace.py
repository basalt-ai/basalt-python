from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from opentelemetry.trace import Span, Tracer

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
    def set_attribute(key: str, value: Any) -> None:
        """Set a raw attribute on the current span."""
        span = get_current_otel_span()
        if span:
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
            new_user = _parse_identity_input(user)
            merged_user = _merge_identity(current_user, new_user)
            if 'id' in merged_user:
                span.set_attribute(semconv.BasaltUser.ID, merged_user['id'])
            if 'name' in merged_user:
                span.set_attribute(semconv.BasaltUser.NAME, merged_user['name'])

        if organization is not None:
            new_org = _parse_identity_input(organization)
            merged_org = _merge_identity(current_org, new_org)
            if 'id' in merged_org:
                span.set_attribute(semconv.BasaltOrganization.ID, merged_org['id'])
            if 'name' in merged_org:
                span.set_attribute(semconv.BasaltOrganization.NAME, merged_org['name'])


def _get_current_identity_from_span(span: Span) -> tuple[dict[str, str], dict[str, str]]:
    """Extract current user and org identity from span attributes."""
    user_dict = {}
    org_dict = {}

    if hasattr(span, 'attributes') and span.attributes:
        if semconv.BasaltUser.ID in span.attributes:
            user_dict['id'] = str(span.attributes[semconv.BasaltUser.ID])
        if semconv.BasaltUser.NAME in span.attributes:
            user_dict['name'] = str(span.attributes[semconv.BasaltUser.NAME])
        if semconv.BasaltOrganization.ID in span.attributes:
            org_dict['id'] = str(span.attributes[semconv.BasaltOrganization.ID])
        if semconv.BasaltOrganization.NAME in span.attributes:
            org_dict['name'] = str(span.attributes[semconv.BasaltOrganization.NAME])

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
        return {'id': value}
    if isinstance(value, dict):
        result = {}
        if 'id' in value:
            result['id'] = str(value['id']) if value['id'] is not None else ''
        if 'name' in value:
            result['name'] = str(value['name']) if value['name'] is not None else ''
        return result
    return {}


def _merge_identity(existing: dict[str, str], new: dict[str, str]) -> dict[str, str]:
    """Merge new identity values into existing, overriding keys present in new."""
    return {**existing, **new}


# Singleton instance
trace_api = Trace
