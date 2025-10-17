"""
Adapter for DatasetsClient to maintain backward compatibility with DatasetSDK.

This adapter wraps the new DatasetsClient and provides the old tuple-based interface.

.. deprecated:: 0.5.0
    The tuple-based API (error, result) is deprecated. Use the new DatasetsClient
    directly which raises exceptions instead of returning error tuples.
    See the migration guide for details.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from ...utils.dtos import (
    CreateDatasetItemResult,
    DatasetDTO,
    DatasetRowDTO,
    GetDatasetResult,
    ListDatasetsResult,
)
from ...utils.protocols import IDatasetSDK

if TYPE_CHECKING:
    from ...datasets.client import DatasetsClient


class DatasetSDKAdapter(IDatasetSDK):
    """
    Adapter that wraps DatasetsClient and provides the old DatasetSDK interface.

    This maintains backward compatibility by converting exceptions back to tuple returns.
    """

    def __init__(self, client: DatasetsClient):
        """
        Initialize the adapter.

        Args:
            client: The new DatasetsClient instance

        .. deprecated:: 0.5.0
            This adapter provides backward compatibility for the tuple-based API.
            Use DatasetsClient directly for the new exception-based API.
        """
        self._client = client
        self._warned = False

    async def list(self) -> ListDatasetsResult:
        """
        Async list datasets with tuple return.

        .. deprecated:: 0.5.0
            Tuple-based error handling is deprecated. Use DatasetsClient.list() directly.

        Returns:
            Tuple[Exception | None, List[DatasetDTO] | None]
        """
        self._emit_deprecation_warning()
        try:
            datasets = await self._client.list()
            # Convert to old DTOs
            dtos = [
                DatasetDTO(
                    slug=ds.slug,
                    name=ds.name,
                    columns=ds.columns,
                    rows=[]
                )
                for ds in datasets
            ]
            return None, dtos
        except Exception as e:
            return e, None

    def list_sync(self) -> ListDatasetsResult:
        """
        Sync list datasets with tuple return.

        .. deprecated:: 0.5.0
            Tuple-based error handling is deprecated. Use DatasetsClient.list_sync() directly.

        Returns:
            Tuple[Exception | None, List[DatasetDTO] | None]
        """
        self._emit_deprecation_warning()
        try:
            datasets = self._client.list_sync()
            # Convert to old DTOs
            dtos = [
                DatasetDTO(
                    slug=ds.slug,
                    name=ds.name,
                    columns=ds.columns,
                    rows=[]
                )
                for ds in datasets
            ]
            return None, dtos
        except Exception as e:
            return e, None

    async def get(self, slug: str) -> GetDatasetResult:
        """
        Async get dataset with tuple return.

        Returns:
            Tuple[Exception | None, DatasetDTO | None]
        """
        try:
            dataset = await self._client.get(slug)
            # Convert to old DTO
            dto = DatasetDTO(
                slug=dataset.slug,
                name=dataset.name,
                columns=dataset.columns,
                rows=[
                    DatasetRowDTO(
                        values=row.values,
                        name=row.name,
                        idealOutput=row.ideal_output,
                        metadata=row.metadata
                    )
                    for row in dataset.rows
                ]
            )
            return None, dto
        except Exception as e:
            return e, None

    def get_sync(self, slug: str) -> GetDatasetResult:
        """
        Sync get dataset with tuple return.

        Returns:
            Tuple[Exception | None, DatasetDTO | None]
        """
        try:
            dataset = self._client.get_sync(slug)
            # Convert to old DTO
            dto = DatasetDTO(
                slug=dataset.slug,
                name=dataset.name,
                columns=dataset.columns,
                rows=[
                    DatasetRowDTO(
                        values=row.values,
                        name=row.name,
                        idealOutput=row.ideal_output,
                        metadata=row.metadata
                    )
                    for row in dataset.rows
                ]
            )
            return None, dto
        except Exception as e:
            return e, None

    async def add_row(
        self,
        slug: str,
        values: dict[str, str],
        name: str | None = None,
        ideal_output: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> CreateDatasetItemResult:
        """
        Async add row with tuple return.

        Returns:
            Tuple[Exception | None, DatasetRowDTO | None, str | None]
        """
        try:
            row, warning = await self._client.add_row(
                slug=slug,
                values=values,
                name=name,
                ideal_output=ideal_output,
                metadata=metadata
            )
            # Convert to old DTO
            dto = DatasetRowDTO(
                values=row.values,
                name=row.name,
                idealOutput=row.ideal_output,
                metadata=row.metadata
            )
            return None, dto, warning
        except Exception as e:
            return e, None, None

    def add_row_sync(
        self,
        slug: str,
        values: dict[str, str],
        name: str | None = None,
        ideal_output: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> CreateDatasetItemResult:
        """
        Sync add row with tuple return.

        Returns:
            Tuple[Exception | None, DatasetRowDTO | None, str | None]
        """
        try:
            row, warning = self._client.add_row_sync(
                slug=slug,
                values=values,
                name=name,
                ideal_output=ideal_output,
                metadata=metadata
            )
            # Convert to old DTO
            dto = DatasetRowDTO(
                values=row.values,
                name=row.name,
                idealOutput=row.ideal_output,
                metadata=row.metadata
            )
            return None, dto, warning
        except Exception as e:
            return e, None, None

    def _emit_deprecation_warning(self):
        """Emit deprecation warning once per adapter instance."""
        if not self._warned:
            warnings.warn(
                "The tuple-based API (error, result) is deprecated and will be removed in v1.0.0. "
                "Use DatasetsClient directly for exception-based error handling. "
                "See https://docs.getbasalt.ai/migration-guide for details.",
                DeprecationWarning,
                stacklevel=3
            )
            self._warned = True
