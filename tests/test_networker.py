import unittest
from unittest.mock import patch, Mock
from parameterized import parameterized

from basalt.utils.networker import Networker
from basalt.utils.errors import NetworkBaseError, FetchError

class TestNetworker(unittest.TestCase):

	@patch('requests.request')
	def test_uses_requests_to_make_http_calls(self, request_mock):
		networker = Networker()

		networker.fetch('http://test/abc', 'GET')

		request_mock.assert_called_once_with('GET', 'http://test/abc', params=None, json=None, headers=None)

	@patch('requests.request')
	def test_captures_requests_exceptions(self, request_mock):
		networker = Networker()
		request_mock.side_effect = Exception('Some unknown error')

		err, res = networker.fetch('http://test/abc', 'GET')

		self.assertIsNone(res)
		self.assertEqual(err.message, 'Some unknown error')
		self.assertIsInstance(err, NetworkBaseError)

	@patch('requests.request')
	def test_rejects_non_json_responses(self, request_mock):
		networker = Networker()
		request_mock.return_value = Mock()
		request_mock.return_value.json.side_effect = Exception('No JSON object could be decoded')

		err, res = networker.fetch('http://test/abc', 'GET')

		self.assertIsNone(res)
		self.assertIsInstance(err, FetchError)

	@patch('requests.request')
	def test_returns_valid_json_as_dict(self, request_mock):
		networker = Networker()
		request_mock.return_value = Mock()
		request_mock.return_value.json.return_value = { "some": "data" }

		err, res = networker.fetch('http://test/abc', 'GET')

		self.assertIsNone(err)
		self.assertEqual(res, { "some": "data" })

	@parameterized.expand([
		(400, 'BadRequest'),
		(401, 'Unauthorized'),
		(403, 'Forbidden'),
		(404, 'NotFound'),
	])
	@patch('requests.request')
	def test_uses_custom_errors(self, response_code, error_type, request_mock):
		networker = Networker()
		request_mock.return_value = Mock()
		request_mock.return_value.status_code = response_code

		err, _ = networker.fetch('http://test/abc', 'GET')

		self.assertIsInstance(err, FetchError)
		self.assertEqual(type(err).__name__, error_type)

