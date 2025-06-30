"""
SDK for interacting with Basalt datasets
"""
from typing import Dict, List, Optional, Tuple, Any

from ..utils.dtos import (
    ListDatasetsDTO, GetDatasetDTO, CreateDatasetItemDTO,
    ListDatasetsResult, GetDatasetResult, CreateDatasetItemResult,
    DatasetDTO, DatasetRowDTO
)
from ..utils.protocols import IApi, ILogger
from ..endpoints.list_datasets import ListDatasetsEndpoint
from ..endpoints.get_dataset import GetDatasetEndpoint
from ..endpoints.create_dataset_item import CreateDatasetItemEndpoint
from ..objects.dataset import Dataset, DatasetRow


class DatasetSDK:
    """
    SDK for interacting with Basalt datasets.
    """
    def __init__(
            self,
            api: IApi,
            logger: ILogger
        ):
        self._api = api
        self._logger = logger

    def list(self) -> ListDatasetsResult:
        """
        List all datasets available in the workspace.

        Returns:
            Tuple[Optional[Exception], Optional[List[DatasetDTO]]]: A tuple containing an optional 
            exception and an optional list of DatasetDTO objects.
        """
        dto = ListDatasetsDTO()
        err, result = self._api.invoke(ListDatasetsEndpoint, dto)

        if err is not None:
            return err, None

        return None, [DatasetDTO(
            slug=dataset.slug,
            name=dataset.name,
            columns=dataset.columns
        ) for dataset in result.datasets]

    def get(self, slug: str) -> GetDatasetResult:
        """
        Get a dataset by its slug.

        Args:
            slug (str): The slug identifier for the dataset.

        Returns:
            Tuple[Optional[Exception], Optional[DatasetDTO]]: A tuple containing an optional
            exception and an optional DatasetDTO.
        """
        dto = GetDatasetDTO(slug=slug)
        err, result = self._api.invoke(GetDatasetEndpoint, dto)

        if err is not None:
            return err, None
            
        if result.error:
            return Exception(result.error), None

        return None, result.dataset

    def addRow(
        self,
        slug: str,
        values: Dict[str, str],
        name: Optional[str] = None,
        ideal_output: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CreateDatasetItemResult:
        """
        Create a new item in a dataset.

        Args:
            slug (str): The slug identifier for the dataset.
            values (Dict[str, str]): A dictionary of column values for the dataset item.
            name (Optional[str]): An optional name for the dataset item.
            ideal_output (Optional[str]): An optional ideal output for the dataset item.
            metadata (Optional[Dict[str, Any]]): An optional metadata dictionary.

        Returns:
            Tuple[Optional[Exception], Optional[DatasetRowDTO], Optional[str]]: A tuple containing
            an optional exception, an optional DatasetRowDTO, and an optional warning message.
        """
        dto = CreateDatasetItemDTO(
            slug=slug,
            values=values,
            name=name,
            idealOutput=ideal_output,
            metadata=metadata
        )

        err, result = self._api.invoke(CreateDatasetItemEndpoint, dto)

        if err is not None:
            return err, None, None
            
        if result.error:
            return Exception(result.error), None, None
            
        return None, result.datasetRow, result.warning
