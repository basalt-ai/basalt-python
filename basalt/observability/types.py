"""Shared observability types."""

from __future__ import annotations

from typing import NotRequired, TypedDict


class _IdentityEntity(TypedDict):
    """Identity entity with required id and optional name."""

    id: str
    name: str


class Identity(TypedDict, total=False):
    """
    Identity structure for user and organization tracking.

    Example:
        {
            "organization": {"id": "123", "name": "ACME"},
            "user": {"id": "456", "name": "John Doe"}
        }
    """

    organization: NotRequired[_IdentityEntity]
    user: NotRequired[_IdentityEntity]
