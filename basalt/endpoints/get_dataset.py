"""
Endpoint for fetching a specific dataset by slug
"""
from dataclasses import dataclass
from typing import Any

from ..utils.dtos import DatasetDTO, GetDatasetDTO


@dataclass
class GetDatasetEndpointResponse:
    """
    Response from the get dataset endpoint
    """
    dataset: DatasetDTO
    error: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GetDatasetEndpointResponse":
        """
        Create an instance of GetDatasetEndpointResponse from a dictionary.

        Args:
            data (Dict[str, Any]): The dictionary containing the response data.

        Returns:
            GetDatasetEndpointResponse
        """
        if "error" in data:
            return cls(dataset=None, error=data["error"])

        return cls(
            dataset=DatasetDTO.from_dict(data["dataset"]),
            error=None
        )


class GetDatasetEndpoint:
    """
    Endpoint class for fetching a specific dataset.
    """

    @staticmethod
    def prepare_request(dto: GetDatasetDTO) -> dict[str, Any]:
        """
        Prepare the request dictionary for the GetDataset endpoint.

        Args:
            dto (GetDatasetDTO): The DTO containing dataset slug.

        Returns:
            The path, method, and query parameters for getting a dataset on the API.
        """
        return {
            "path": f"/datasets/{dto.slug}",
            "method": "GET",
            "query": {}
        }

    @staticmethod
    def decode_response(response: dict) -> tuple[Exception | None, GetDatasetEndpointResponse | None]:
        """
        Decode the response returned from the API

        Args:
            response (dict): The JSON response to encode into a GetDatasetEndpointResponse

        Returns:
            A tuple containing an optional exception and an optional GetDatasetEndpointResponse.
        """
        try:
            return None, GetDatasetEndpointResponse.from_dict(response)
        except Exception as e:
            return e, None
