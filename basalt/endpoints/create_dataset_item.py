"""
Endpoint for creating a new dataset item
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ..utils.dtos import DatasetRowDTO, CreateDatasetItemDTO

@dataclass
class CreateDatasetItemEndpointResponse:
    """
    Response from the create dataset item endpoint
    """
    datasetRow: DatasetRowDTO
    warning: Optional[str] = None
    error: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CreateDatasetItemEndpointResponse":
        """
        Create an instance of CreateDatasetItemEndpointResponse from a dictionary.

        Args:
            data (Dict[str, Any]): The dictionary containing the response data.

        Returns:
            CreateDatasetItemEndpointResponse
        """
        if "error" in data:
            return cls(datasetRow=None, error=data["error"])
            
        return cls(
            datasetRow=DatasetRowDTO.from_dict(data["datasetRow"]),
            warning=data.get("warning"),
            error=None
        )


class CreateDatasetItemEndpoint:
    """
    Endpoint class for creating a dataset item.
    """
    @staticmethod
    def prepare_request(dto: CreateDatasetItemDTO) -> Dict[str, Any]:
        """
        Prepare the request dictionary for the CreateDatasetItem endpoint.

        Args:
            dto (CreateDatasetItemDTO): The DTO containing dataset item data.
            
        Returns:
            The path, method, and body for creating a dataset item on the API.
        """
        body = {
            "values": dto.values
        }
        
        if dto.name:
            body["name"] = dto.name
            
        if dto.idealOutput:
            body["idealOutput"] = dto.idealOutput
            
        if dto.metadata:
            body["metadata"] = dto.metadata
        
        return {
            "path": f"/datasets/{dto.slug}/items",
            "method": "POST",
            "body": body
        }

    @staticmethod
    def decode_response(response: dict) -> Tuple[Optional[Exception], Optional[CreateDatasetItemEndpointResponse]]:
        """
        Decode the response returned from the API

        Args:
            response (dict): The JSON response to encode into a CreateDatasetItemEndpointResponse

        Returns:
            A tuple containing an optional exception and an optional CreateDatasetItemEndpointResponse.
        """
        try:
            return None, CreateDatasetItemEndpointResponse.from_dict(response)
        except Exception as e:
            return e, None
