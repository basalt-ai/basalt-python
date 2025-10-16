from dataclasses import dataclass
from typing import Any

from ..utils.dtos import DescribePromptDTO, DescribePromptResponse


@dataclass
class DescribePromptEndpointResponse:
    warning: str | None
    prompt: DescribePromptResponse

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DescribePromptEndpointResponse":
        """
        Create an instance of DescribePromptEndpointResponse from a dictionary.

        Args:
            data (Dict[str, Any]): The dictionary containing the response data.

        Returns:
            DescribePromptEndpointResponse
        """
        return cls(
            warning=data.get("warning"),
            prompt=DescribePromptResponse.from_dict(data["prompt"]),
        )


class DescribePromptEndpoint:
    """
    Endpoint class for fetching a prompt.
    """

    @staticmethod
    def prepare_request(dto: DescribePromptDTO) -> dict[str, Any]:
        """
        Prepare the request dictionary for the DescribePrompt endpoint.

        Args:
            dto (DescribePromptDTO): The data transfer object containing the request parameters.

        Returns:
        	The path, method, and query parameters for describing a prompt on the API.
        """
        return {
            "path": f"/prompts/{dto.slug}/describe",
            "method": "GET",
            "query": {
                "version": dto.version,
                "tag": dto.tag
            }
        }

    @staticmethod
    def decode_response(response: dict) -> tuple[Exception | None, DescribePromptEndpointResponse | None]:
        """
        Decode the response returned from the API

        Args:
            response (dict): The JSON response to encode into a DescribePromptEndpointResponse

        Returns:
        	A tuple containing an optional exception and an optional DescribePromptEndpointResponse.
        """
        try:
            return None, DescribePromptEndpointResponse.from_dict(response)
        except Exception as e:
            return e, None
