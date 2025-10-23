"""Tests for the HTTP client."""
from unittest.mock import Mock, patch

import pytest

from basalt._internal.exceptions import (
    BadRequestError,
    ForbiddenError,
    NetworkError,
    NotFoundError,
    UnauthorizedError,
)
from basalt._internal.http import HTTPClient


# Helper function to create a mock session
def create_mock_session():
    """Create a mock session for testing."""
    session = Mock()
    return session


class TestHTTPClient:
    """Test cases for the HTTPClient class."""

    @patch('basalt._internal.http.requests.Session.request')
    def test_uses_requests_to_make_http_calls(self, request_mock):
        """Test that the client uses requests library for HTTP calls."""
        client = HTTPClient()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'application/json'}
        mock_response.json.return_value = {}
        request_mock.return_value = mock_response

        client.fetch_sync('http://test/abc', 'GET')

        # Verify the request was called with expected parameters
        assert request_mock.call_count == 1
        call_args = request_mock.call_args
        assert call_args[0][0] == 'GET'
        assert call_args[0][1] == 'http://test/abc'

    @patch('basalt._internal.http.requests.Session.request')
    def test_captures_requests_exceptions(self, request_mock):
        """Test that the client captures and wraps requests exceptions."""
        client = HTTPClient()
        request_mock.side_effect = Exception('Some unknown error')

        with pytest.raises(NetworkError) as exc_info:
            client.fetch_sync('http://test/abc', 'GET')

        assert 'Some unknown error' in str(exc_info.value.message)

    @patch('basalt._internal.http.requests.Session.request')
    def test_rejects_non_json_responses(self, request_mock):
        """Test that the client handles non-JSON responses properly."""
        client = HTTPClient()
        request_mock.return_value = Mock()
        request_mock.return_value.json.side_effect = Exception('No JSON object could be decoded')
        request_mock.return_value.headers = {}
        request_mock.return_value.status_code = 200

        with pytest.raises(NetworkError):
            client.fetch_sync('http://test/abc', 'GET')

    @patch('basalt._internal.http.requests.Session.request')
    def test_returns_valid_json_as_dict(self, request_mock):
        """Test that the client returns valid JSON responses."""
        client = HTTPClient()
        mock_response = Mock()
        mock_response.json.return_value = {"some": "data"}
        mock_response.headers = {'Content-Type': 'application/json'}
        mock_response.status_code = 200
        request_mock.return_value = mock_response

        result = client.fetch_sync('http://test/abc', 'GET')

        assert result == {"some": "data"}

    @pytest.mark.parametrize("response_code,error_class", [
        (400, BadRequestError),
        (401, UnauthorizedError),
        (403, ForbiddenError),
        (404, NotFoundError),
    ])
    @patch('basalt._internal.http.requests.Session.request')
    def test_raises_custom_errors(self, request_mock, response_code, error_class):
        """Test that the client raises appropriate custom errors for HTTP error codes."""
        client = HTTPClient()
        mock_response = Mock()
        mock_response.status_code = response_code
        mock_response.headers = {'Content-Type': 'application/json'}
        mock_response.json.return_value = {}
        mock_response.text = ""
        request_mock.return_value = mock_response

        with pytest.raises(error_class):
            client.fetch_sync('http://test/abc', 'GET')

    @patch('basalt._internal.http.requests.Session.request')
    def test_includes_error_message_from_api(self, request_mock):
        """Test that the client includes error messages from the API response."""
        client = HTTPClient()
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.headers = {'Content-Type': 'application/json'}
        mock_response.json.return_value = {'error': 'Invalid request format'}
        mock_response.text = ""
        request_mock.return_value = mock_response

        with pytest.raises(BadRequestError) as exc_info:
            client.fetch_sync('http://test/abc', 'GET')

        assert exc_info.value.message == 'Invalid request format'
        assert exc_info.value.status_code == 400

    @patch('basalt._internal.http.requests.Session.request')
    def test_handles_errors_field_for_bad_request(self, request_mock):
        """Test that the client handles 'errors' field in bad request responses."""
        client = HTTPClient()
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.headers = {'Content-Type': 'application/json'}
        mock_response.json.return_value = {'errors': 'Validation failed'}
        mock_response.text = ""
        request_mock.return_value = mock_response

        with pytest.raises(BadRequestError) as exc_info:
            client.fetch_sync('http://test/abc', 'GET')

        assert exc_info.value.message == 'Validation failed'

    @patch('basalt._internal.http.requests.Session.request')
    def test_handles_202_no_content(self, request_mock):
        """Test that the client handles 202 Accepted responses with no content."""
        client = HTTPClient()
        mock_response = Mock()
        mock_response.status_code = 202
        mock_response.headers = {}
        mock_response.json.side_effect = Exception('No content')
        request_mock.return_value = mock_response

        result = client.fetch_sync('http://test/abc', 'POST')

        assert result is None

    @patch('basalt._internal.http.requests.Session.request')
    def test_handles_204_no_content(self, request_mock):
        """Test that the client handles 204 No Content responses."""
        client = HTTPClient()
        mock_response = Mock()
        mock_response.status_code = 204
        mock_response.headers = {}
        mock_response.json.side_effect = Exception('No content')
        request_mock.return_value = mock_response

        result = client.fetch_sync('http://test/abc', 'DELETE')

        assert result is None

    @pytest.mark.parametrize("method", ["GET", "POST", "PUT", "DELETE"])
    @patch('basalt._internal.http.requests.Session.request')
    def test_supports_http_methods(self, request_mock, method):
        """Test that the client supports various HTTP methods."""
        client = HTTPClient()
        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_response.headers = {'Content-Type': 'application/json'}
        mock_response.status_code = 200
        request_mock.return_value = mock_response

        client.fetch_sync('http://test/abc', method)

        call_args = request_mock.call_args[0]
        assert call_args[0] == method

    @patch('basalt._internal.http.requests.Session.request')
    def test_passes_body_to_request(self, request_mock):
        """Test that the client passes request body correctly."""
        client = HTTPClient()
        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_response.headers = {'Content-Type': 'application/json'}
        mock_response.status_code = 200
        request_mock.return_value = mock_response

        client.fetch_sync('http://test/abc', 'POST', body={"test": "data"})

        call_kwargs = request_mock.call_args.kwargs
        assert call_kwargs['json'] == {"test": "data"}

    @patch('basalt._internal.http.requests.Session.request')
    def test_passes_params_to_request(self, request_mock):
        """Test that the client passes query parameters correctly."""
        client = HTTPClient()
        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_response.headers = {'Content-Type': 'application/json'}
        mock_response.status_code = 200
        request_mock.return_value = mock_response

        client.fetch_sync('http://test/abc', 'GET', params={"tag": "test"})

        call_kwargs = request_mock.call_args.kwargs
        assert call_kwargs['params'] == {"tag": "test"}

    @patch('basalt._internal.http.requests.Session.request')
    def test_passes_headers_to_request(self, request_mock):
        """Test that the client passes headers correctly."""
        client = HTTPClient()
        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_response.headers = {'Content-Type': 'application/json'}
        mock_response.status_code = 200
        request_mock.return_value = mock_response

        client.fetch_sync('http://test/abc', 'GET', headers={"Authorization": "Bearer token"})

        call_kwargs = request_mock.call_args.kwargs
        assert call_kwargs['headers'] == {"Authorization": "Bearer token"}

    def test_context_manager_sync(self):
        """Test that the client supports sync context manager protocol."""
        with HTTPClient() as client:
            assert client is not None
            assert client._sync_session is None

    @pytest.mark.asyncio
    async def test_context_manager_async(self):
        """Test that the client supports async context manager protocol."""
        async with HTTPClient() as client:
            assert client is not None
            assert client._session is None

    def test_ssl_verification_disabled(self):
        """Test that SSL verification can be disabled."""
        client = HTTPClient(verify_ssl=False)
        assert client.verify_ssl is False

    def test_custom_timeout(self):
        """Test that custom timeout can be set."""
        client = HTTPClient(timeout=60.0)
        assert client.timeout == 60.0

    def test_custom_retries(self):
        """Test that custom retry count can be set."""
        client = HTTPClient(max_retries=5)
        assert client.max_retries == 5
