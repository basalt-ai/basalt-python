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


class TestHTTPClient:
    """Test cases for the HTTPClient class."""

    @patch('requests.request')
    def test_uses_requests_to_make_http_calls(self, request_mock):
        """Test that the client uses requests library for HTTP calls."""
        client = HTTPClient()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'application/json'}
        mock_response.json.return_value = {}
        request_mock.return_value = mock_response

        client.fetch_sync('http://test/abc', 'GET')

        request_mock.assert_called_once_with('GET', 'http://test/abc', params=None, json=None, headers=None)

    @patch('requests.request')
    def test_captures_requests_exceptions(self, request_mock):
        """Test that the client captures and wraps requests exceptions."""
        client = HTTPClient()
        request_mock.side_effect = Exception('Some unknown error')

        with pytest.raises(NetworkError) as exc_info:
            client.fetch_sync('http://test/abc', 'GET')

        assert 'Some unknown error' in str(exc_info.value.message)

    @patch('requests.request')
    def test_rejects_non_json_responses(self, request_mock):
        """Test that the client handles non-JSON responses properly."""
        client = HTTPClient()
        request_mock.return_value = Mock()
        request_mock.return_value.json.side_effect = Exception('No JSON object could be decoded')
        request_mock.return_value.headers = {}
        request_mock.return_value.status_code = 200

        with pytest.raises(NetworkError):
            client.fetch_sync('http://test/abc', 'GET')

    @patch('requests.request')
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
    @patch('requests.request')
    def test_raises_custom_errors(self, request_mock, response_code, error_class):
        """Test that the client raises appropriate custom errors for HTTP error codes."""
        client = HTTPClient()
        mock_response = Mock()
        mock_response.status_code = response_code
        mock_response.headers = {'Content-Type': 'application/json'}
        mock_response.json.return_value = {}
        request_mock.return_value = mock_response

        with pytest.raises(error_class):
            client.fetch_sync('http://test/abc', 'GET')

    @patch('requests.request')
    def test_includes_error_message_from_api(self, request_mock):
        """Test that the client includes error messages from the API response."""
        client = HTTPClient()
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.headers = {'Content-Type': 'application/json'}
        mock_response.json.return_value = {'error': 'Invalid request format'}
        request_mock.return_value = mock_response

        with pytest.raises(BadRequestError) as exc_info:
            client.fetch_sync('http://test/abc', 'GET')

        assert exc_info.value.message == 'Invalid request format'
        assert exc_info.value.status_code == 400

    @patch('requests.request')
    def test_handles_errors_field_for_bad_request(self, request_mock):
        """Test that the client handles 'errors' field in bad request responses."""
        client = HTTPClient()
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.headers = {'Content-Type': 'application/json'}
        mock_response.json.return_value = {'errors': 'Validation failed'}
        request_mock.return_value = mock_response

        with pytest.raises(BadRequestError) as exc_info:
            client.fetch_sync('http://test/abc', 'GET')

        assert exc_info.value.message == 'Validation failed'

    @patch('requests.request')
    def test_handles_202_no_content(self, request_mock):
        """Test that the client handles 202 Accepted responses with no content."""
        client = HTTPClient()
        mock_response = Mock()
        mock_response.status_code = 202
        mock_response.headers = {}
        mock_response.json.side_effect = Exception('No content')
        request_mock.return_value = mock_response

        result = client.fetch_sync('http://test/abc', 'POST')

        assert result == {}

    @patch('requests.request')
    def test_handles_204_no_content(self, request_mock):
        """Test that the client handles 204 No Content responses."""
        client = HTTPClient()
        mock_response = Mock()
        mock_response.status_code = 204
        mock_response.headers = {}
        mock_response.json.side_effect = Exception('No content')
        request_mock.return_value = mock_response

        result = client.fetch_sync('http://test/abc', 'DELETE')

        assert result == {}

    @pytest.mark.parametrize("method", ["GET", "POST", "PUT", "DELETE"])
    @patch('requests.request')
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

    @patch('requests.request')
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

    @patch('requests.request')
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

    @patch('requests.request')
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
