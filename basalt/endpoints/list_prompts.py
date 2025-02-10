from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

from ..utils.dtos import PromptListResponse

@dataclass
class ListPromptsEndpointResponse:
    warning: Optional[str]
    prompts: List[PromptListResponse]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ListPromptsEndpointResponse":
        """
        Create an instance of ListPromptsEndpointResponse from a dictionary.

        Args:
            data (Dict[str, Any]): The dictionary containing the response data.

        Returns:
            ListPromptsEndpointResponse
        """
        return cls(
            warning=data.get("warning"),
            prompts=[PromptListResponse.from_dict(prompt) for prompt in data["prompts"]],
        )

class ListPromptsEndpoint:
    """
    Endpoint class for fetching a prompt.
    """
    @staticmethod
    def prepare_request() -> Dict[str, Any]:
        """
        Prepare the request dictionary for the ListPrompts endpoint.

        Returns:
        	The path, method, and query parameters for getting a prompt on the API.
        """
        return {
            "path": "/prompts",
            "method": "GET"
        }

    @staticmethod
    def decode_response(response: dict) -> Tuple[Optional[Exception], Optional[ListPromptsEndpointResponse]]:
        """
        Decode the response returned from the API

        Args:
            response (dict): The JSON response to encode into a ListPromptsEndpointResponse

        Returns:
        	A tuple containing an optional exception and an optional ListPromptsEndpointResponse.
        """
        try:
            return None, ListPromptsEndpointResponse.from_dict(response)
        except Exception as e:
            return e, None
