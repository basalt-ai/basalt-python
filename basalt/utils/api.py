from typing import Dict, TypeVar, Optional, Tuple

from .protocols import IEndpoint, INetworker, ILogger
from .networker import Networker

Input = TypeVar('Input')
Output = TypeVar('Output')

class Api:
    """
    A class to interact with the Basalt API.
    
    Attributes:
        root_url (str): The root URL of the API.
        api_key (str): The API key for authentication.
        sdk_version (str): The version of the SDK.
        sdk_type (str): The SDK type (ex: py-pip)
        networker (INetworker): The networker instance to handle network requests.
    """
    def __init__(self, root_url: str, networker: INetworker, api_key: str, sdk_version: str, sdk_type: str, logger: Optional[ILogger] = None):
        """
        Initialize the Api class with the given parameters.
        
        Args:
            root_url (str): The root URL of the API.
            networker (INetworker): The networker instance to handle network requests.
            api_key (str): The API key for authentication.
            sdk_version (str): The version of the SDK.
            sdk_type (str): The SDK type (ex: py-pip)
            logger (Optional[ILogger]): The logger instance for debug messages.
        """
        self._root = root_url
        self._api_key = api_key
        self._sdk_version = sdk_version
        self._sdk_type = sdk_type
        self._logger = logger
        self._network = networker
        if isinstance(networker, Networker):
            networker._logger = logger

    def invoke(
        self,
        endpoint: IEndpoint[Input, Output],
        dto: Optional[Input] = None
    ) -> Tuple[Optional[Exception], Optional[Output]]:
        """
        Invoke an API endpoint with the given data transfer object (DTO).

        Args:
            endpoint: The endpoint to be invoked.
            dto: The data transfer object to be sent to the endpoint.

        Returns:
            A tuple containing an optional exception and an optional output.
        """
        # Prepare the request information using the endpoint and input data
        if dto is None:
            request_info = endpoint.prepare_request()
        else:
            request_info = endpoint.prepare_request(dto)

        # Fetch the result from the network using the prepared request information
        error, result = self._network.fetch(
            self._root + request_info['path'],
            request_info['method'],
            request_info.get('body'),
            params=request_info.get('query', {}),
            headers=self._headers(),
        )

        if error:
            return error, None

        return endpoint.decode_response(result)

    def _headers(self) -> Dict[str, str]:
        """
        Generate headers for the request including authorization and SDK information.
        """
        return {
            'Authorization': f'Bearer {self._api_key}',
            'X-BASALT-SDK-VERSION': self._sdk_version,
            'X-BASALT-SDK-TYPE': self._sdk_type,
            'Content-Type': 'application/json'
        }
