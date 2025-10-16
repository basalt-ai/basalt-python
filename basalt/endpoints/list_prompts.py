from dataclasses import dataclass
from typing import Any

from ..utils.dtos import PromptListDTO, PromptListResponse


@dataclass
class ListPromptsEndpointResponse:
    warning: str | None
    prompts: list[PromptListResponse]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ListPromptsEndpointResponse":
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
    def prepare_request(dto: PromptListDTO) -> dict[str, Any]:
        """
        Prepare the request dictionary for the ListPrompts endpoint.

        Returns:
        	The path, method, and query parameters for getting a prompt on the API.
        """
        return {
            "path": "/prompts",
            "method": "GET",
            "query": {
                "featureSlug": dto.featureSlug
            }
        }

    @staticmethod
    def decode_response(response: dict) -> tuple[Exception | None, ListPromptsEndpointResponse | None]:
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
