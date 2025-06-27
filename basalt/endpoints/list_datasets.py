"""
Endpoint for listing all datasets
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..utils.dtos import DatasetDTO, ListDatasetsDTO

@dataclass
class ListDatasetsEndpointResponse:
    """
    Response from the list datasets endpoint
    """
    datasets: List[DatasetDTO]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ListDatasetsEndpointResponse":
        """
        Create an instance of ListDatasetsEndpointResponse from a dictionary.

        Args:
            data (Dict[str, Any]): The dictionary containing the response data.

        Returns:
            ListDatasetsEndpointResponse
        """
        return cls(
            datasets=[DatasetDTO.from_dict(dataset) for dataset in data["datasets"]],
        )


class ListDatasetsEndpoint:
    """
    Endpoint class for fetching all datasets.
    """
    @staticmethod
    def prepare_request(dto: ListDatasetsDTO) -> Dict[str, Any]:
        """
        Prepare the request dictionary for the ListDatasets endpoint.

        Returns:
            The path, method, and query parameters for getting datasets on the API.
        """
        return {
            "path": "/datasets",
            "method": "GET",
            "query": {}
        }

    @staticmethod
    def decode_response(response: dict) -> Tuple[Optional[Exception], Optional[ListDatasetsEndpointResponse]]:
        """
        Decode the response returned from the API

        Args:
            response (dict): The JSON response to encode into a ListDatasetsEndpointResponse

        Returns:
            A tuple containing an optional exception and an optional ListDatasetsEndpointResponse.
        """
        try:
            return None, ListDatasetsEndpointResponse.from_dict(response)
        except Exception as e:
            return e, None
