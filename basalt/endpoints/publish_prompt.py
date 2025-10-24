"""
Endpoint for publishing a prompt with a tag
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ..utils.dtos import DeploymentTagResponse, PublishPromptDTO


@dataclass
class PublishPromptEndpointResponse:
	"""
	Response from the publish prompt endpoint
	"""
	deploymentTag: DeploymentTagResponse
	error: Optional[str] = None

	@classmethod
	def from_dict(cls, data: Dict[str, Any]) -> "PublishPromptEndpointResponse":
		"""
		Create an instance of PublishPromptEndpointResponse from a dictionary.

		Args:
			data (Dict[str, Any]): The dictionary containing the response data.

		Returns:
			PublishPromptEndpointResponse
		"""
		if "error" in data:
			return cls(deploymentTag=None, error=data["error"])

		return cls(
			deploymentTag=DeploymentTagResponse.from_dict(data["deploymentTag"]),
			error=None
		)


class PublishPromptEndpoint:
	"""
	Endpoint class for publishing a prompt with a tag.
	"""
	@staticmethod
	def prepare_request(dto: PublishPromptDTO) -> Dict[str, Any]:
		"""
		Prepare the request dictionary for the PublishPrompt endpoint.

		Args:
			dto (PublishPromptDTO): The DTO containing publish prompt data.

		Returns:
			The path, method, and body for publishing a prompt on the API.
		"""
		body = {
			"newTag": dto.new_tag
		}

		if dto.version:
			body["version"] = dto.version

		if dto.tag:
			body["tag"] = dto.tag

		return {
			"path": f"/prompts/{dto.slug}/publish",
			"method": "POST",
			"body": body
		}

	@staticmethod
	def decode_response(
		response: dict
	) -> Tuple[Optional[Exception], Optional[PublishPromptEndpointResponse]]:
		"""
		Decode the response returned from the API

		Args:
			response (dict): The JSON response to encode into a PublishPromptEndpointResponse

		Returns:
			A tuple containing an optional exception and an optional PublishPromptEndpointResponse.
		"""
		try:
			return None, PublishPromptEndpointResponse.from_dict(response)
		except Exception as e:
			return e, None
