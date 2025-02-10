from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ..utils.dtos import GetPromptDTO, PromptResponse

@dataclass
class GetPromptEndpointResponse:
    warning: Optional[str]
    prompt: PromptResponse

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GetPromptEndpointResponse":
        """
        Create an instance of GetPromptEndpointResponse from a dictionary.

        Args:
            data (Dict[str, Any]): The dictionary containing the response data.

        Returns:
            GetPromptEndpointResponse
        """
        return cls(
            warning=data.get("warning"),
            prompt=PromptResponse.from_dict(data["prompt"]),
        )

class GetPromptEndpoint:
    """
    Endpoint class for fetching a prompt.
    """
    @staticmethod
    def prepare_request(dto: GetPromptDTO) -> Dict[str, Any]:
        """
        Prepare the request dictionary for the GetPrompt endpoint.

        Args:
            dto (GetPromptDTO): The data transfer object containing the request parameters.

        Returns:
        	The path, method, and query parameters for getting a prompt on the API.
        """
        return {
            "path": f"/prompts/{dto.slug}",
            "method": "GET",
            "query": {
                "version": dto.version,
                "tag": dto.tag
            }
        }

    @staticmethod
    def decode_response(response: dict) -> Tuple[Optional[Exception], Optional[GetPromptEndpointResponse]]:
        """
        Decode the response returned from the API

        Args:
            response (dict): The JSON response to encode into a GetPromptEndpointResponse

        Returns:
        	A tuple containing an optional exception and an optional GetPromptEndpointResponse.
        """
        try:
            return None, GetPromptEndpointResponse.from_dict(response)
        except Exception as e:
            return e, None
