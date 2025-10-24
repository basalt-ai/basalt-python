import requests
import aiohttp
from typing import Any, Dict, Optional, Tuple, Mapping

from .errors import BadRequest, FetchError, Forbidden, NetworkBaseError, NotFound, Unauthorized, UnprocessableEntity
from .protocols import INetworker

class Networker(INetworker):
    """
    Networker class that implements the INetworker protocol.
    Provides a method to fetch data from a given URL using HTTP methods.
    """
    def __init__(self):
        pass

    async def fetch(
            self,
            url: str,
            method: str,
            body: Optional[Any] = None,
            params: Optional[Mapping[str, str]] = None,
            headers: Optional[Mapping[str, str]] = None
        ) -> Tuple[Optional[FetchError], Optional[Dict[str, Any]]]:
        """
        Fetch data from a given URL using the specified HTTP method.

        Args:
            url (str): The URL to fetch data from.
            method (str): The HTTP method to use (e.g., 'GET', 'POST').
            body (Optional[Any]): The request payload to send (default is None).
            headers (Optional[Dict[str, str]]): The request headers to send (default is None).
            params (Optional[Dict[str, str]]): The query parameters to send (default is None).

        Returns:
            A result tuple (err, json_response), possible responses:
            - (None, json_response)
            - (FetchError, None)
        """
        try:
            # Filter out None values from params and headers
            filtered_params = {k: v for k, v in params.items() if v is not None} if params else None
            filtered_headers = {k: v for k, v in headers.items() if v is not None} if headers else None

            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
                async with session.request(
                    method,
                    url,
                    params=filtered_params,
                    json=body,
                    headers=filtered_headers
                ) as response:
                    # Try to parse JSON response, but handle cases where there's no JSON body
                    json_response = None
                    content_type = response.headers.get('Content-Type', '')
                    if content_type and 'application/json' in content_type:
                        try:
                            json_response = await response.json()
                        except Exception:
                            json_response = {}
                    elif response.status not in [202, 204]:
                        # For non-202/204 responses without JSON content-type, try to parse anyway
                        try:
                            json_response = await response.json()
                        except Exception:
                            json_response = {}
                    else:
                        # For 202 (Accepted) or 204 (No Content), an empty body is expected
                        json_response = {}

                    if response.status == 400:
                        return BadRequest(json_response.get('error', json_response.get('errors', 'Bad Request'))), None

                    if response.status == 401:
                        return Unauthorized(json_response.get('error', 'Unauthorized')), None

                    if response.status == 403:
                        return Forbidden(json_response.get('error', 'Forbidden')), None

                    if response.status == 404:
                        return NotFound(json_response.get('error', 'Not Found')), None

                    if response.status == 422:
                        return UnprocessableEntity(json_response.get('error', 'Unprocessable Entity')), None

                    response.raise_for_status()

                    return None, json_response

        except Exception as e:
            return NetworkBaseError(str(e)), None

    def fetch_sync(
            self,
            url: str,
            method: str,
            body: Optional[Any] = None,
            params: Optional[Mapping[str, str]] = None,
            headers: Optional[Mapping[str, str]] = None
        ) -> Tuple[Optional[FetchError], Optional[Dict[str, Any]]]:
        """
        Synchronously fetch data from a given URL using the specified HTTP method. This method should never throw.

        Args:
            url (str): The URL to fetch data from.
            method (str): The HTTP method to use (e.g., 'GET', 'POST').
            body (Optional[Any]): The request payload to send (default is None).
            headers (Optional[Dict[str, str]]): The request headers to send (default is None).
            params (Optional[Dict[str, str]]): The query parameters to send (default is None).

        Returns:
            A result tuple (err, json_response), possible responses:
            - (None, json_response)
            - (FetchError, None)
        """
        try:
            response = requests.request(
                method,
                url,
                params=params,
                json=body,
                headers=headers
            )

            # Try to parse JSON response, but handle cases where there's no JSON body
            json_response = None
            content_type = response.headers.get('Content-Type', '')
            if content_type and 'application/json' in content_type:
                try:
                    json_response = response.json()
                except Exception:
                    json_response = {}
            elif response.status_code not in [202, 204]:
                # For non-202/204 responses without JSON content-type, try to parse anyway
                try:
                    json_response = response.json()
                except Exception:
                    json_response = {}
            else:
                # For 202 (Accepted) or 204 (No Content), an empty body is expected
                json_response = {}

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
