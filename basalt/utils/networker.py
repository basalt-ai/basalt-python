import requests
from typing import Any, Dict, Optional, Tuple

from .errors import BadRequest, FetchError, Forbidden, NetworkBaseError, NotFound, Unauthorized
from .protocols import INetworker, ILogger

class Networker(INetworker):
    """
    Networker class that implements the INetworker protocol.
    Provides a method to fetch data from a given URL using HTTP methods.
    """
    def __init__(self):
        pass

    def fetch(
            self,
            url: str,
            method: str,
            body = None,
            headers = None,
            params = None
        ) -> Tuple[Optional[FetchError], Optional[Dict[str, Any]]]:
        """
            Performs an HTTP request and returns either a parsed JSON response or a FetchError.
            
            Sends a request to the specified URL using the given HTTP method, with optional body, headers, and query parameters. Returns a tuple where the first element is a FetchError instance on failure (with the second element as None), or None on success (with the second element as the parsed JSON response). This method never raises exceptions.
             
            Args:
                url: The endpoint to send the request to.
                method: The HTTP method to use (e.g., 'GET', 'POST').
                body: Optional request payload.
                headers: Optional dictionary of request headers.
                params: Optional dictionary of query parameters.
            
            Returns:
                A tuple (error, json_response):
                    - (None, json_response) on success.
                    - (FetchError, None) on failure.
            """
        try:
            response = requests.request(
                method,
                url,
                params=params,
                json=body,
                headers=headers
            )

            json_response = response.json()

            if response.status_code == 400:
                return BadRequest(json_response.get('error', json_response.get('errors', 'Bad Request'))), None

            if response.status_code == 401:
                return Unauthorized(json_response.get('error', 'Unauthorized')), None

            if response.status_code == 403:
                return Forbidden(json_response.get('error', 'Forbidden')), None

            if response.status_code == 404:
                return NotFound(json_response.get('error', 'Not Found')), None

            response.raise_for_status()

            return None, json_response

        except Exception as e:
            return NetworkBaseError(str(e)), None
