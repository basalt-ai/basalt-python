import unittest
from unittest.mock import MagicMock
from parameterized import parameterized

from basalt.utils.api import Api

class TestApi(unittest.TestCase):

	def test_uses_endpoint_to_encode_request(self):
		mocked_network = MagicMock()
		mocked_network.fetch.return_value = (None, {})

		mocked_endpoint = MagicMock()
		mocked_endpoint.prepare_request.return_value = {
			"path": "/some-path",
			"method": "GET",
			"query": { "tag": "abc" }
		}
		mocked_endpoint.decode_response.return_value = (None, {})

		api = Api(
			networker=mocked_network,
			root_url="https://basalt-test/",
			api_key="my-api-key",
			sdk_version="1.0.0",
			sdk_type="py-test"
		)

		api.invoke(mocked_endpoint, { "some": "dto" })

		mocked_endpoint.prepare_request.assert_called_once_with({ "some": "dto" })

	def test_uses_endpoint_to_decode_response(self):
		mocked_network = MagicMock()
		mocked_network.fetch.return_value = (None, { "some": "response" })

		mocked_endpoint = MagicMock()
		mocked_endpoint.prepare_request.return_value = {
			"path": "/some-path",
			"method": "GET",
			"query": { "tag": "abc" }
		}
		mocked_endpoint.decode_response.return_value = (None, { "decoded": "response" })

		api = Api(
			networker=mocked_network,
			root_url="https://basalt-test/",
			api_key="my-api-key",
			sdk_version="1.0.0",
			sdk_type="py-test"
		)

		err, res = api.invoke(mocked_endpoint, { "some": "dto" })

		mocked_endpoint.decode_response.assert_called_once_with({ "some": "response" })

		self.assertIsNone(err)
		self.assertEqual(res, { "decoded": "response" })

	def test_forwards_decoder_error(self):
		mocked_network = MagicMock()
		mocked_network.fetch.return_value = (None, { "some": "response" })

		mocked_endpoint = MagicMock()
		mocked_endpoint.prepare_request.return_value = {
			"path": "/some-path",
			"method": "GET",
			"query": { "tag": "abc" }
		}
		mocked_endpoint.decode_response.return_value = (Exception("Bad response format"), None)

		api = Api(
			networker=mocked_network,
			root_url="https://basalt-test/",
			api_key="my-api-key",
			sdk_version="1.0.0",
			sdk_type="py-test"
		)

		err, res = api.invoke(mocked_endpoint, { "some": "dto" })

		mocked_endpoint.decode_response.assert_called_once_with({ "some": "response" })

		self.assertIsNone(res)
		self.assertIsInstance(err, Exception)
		self.assertEqual(str(err), "Bad response format")

	@parameterized.expand(["GET", "POST", "PUT", "DELETE"])
	def test_uses_http_verb(self, http_verb):
		mocked_network = MagicMock()
		mocked_network.fetch.return_value = (None, { "some": "response" })

		mocked_endpoint = MagicMock()
		mocked_endpoint.prepare_request.return_value = {
			"path": "/",
			"method": http_verb,
			"query": {}
		}
		mocked_endpoint.decode_response.return_value = (None, { "some": "response" })

		api = Api(
			networker=mocked_network,
			root_url="https://basalt-test/",
			api_key="my-api-key",
			sdk_version="1.0.0",
			sdk_type="py-test"
		)

		api.invoke(mocked_endpoint, { "some": "dto" })

		call_args = mocked_network.fetch.call_args[0]

		self.assertEqual(call_args[1], http_verb)

	def test_prefixes_api_root(self):
		mocked_network = MagicMock()
		mocked_network.fetch.return_value = (None, { "some": "response" })

		mocked_endpoint = MagicMock()
		mocked_endpoint.prepare_request.return_value = {
			"path": "/test-path",
			"method": "GET",
			"query": {}
		}
		mocked_endpoint.decode_response.return_value = (None, { "some": "response" })

		api = Api(
			networker=mocked_network,
			root_url="https://basalt-test/",
			api_key="my-api-key",
			sdk_version="1.0.0",
			sdk_type="py-test"
		)

		api.invoke(mocked_endpoint, { "some": "dto" })

		call_args = mocked_network.fetch.call_args[0]

		self.assertTrue(call_args[0].startswith("https://basalt-test/"))

	def test_includes_path_in_url(self):
		mocked_network = MagicMock()
		mocked_network.fetch.return_value = (None, { "some": "response" })

		mocked_endpoint = MagicMock()
		mocked_endpoint.prepare_request.return_value = {
			"path": "/test-path",
			"method": "GET",
			"query": {}
		}
		mocked_endpoint.decode_response.return_value = (None, { "some": "response" })

		api = Api(
			networker=mocked_network,
			root_url="https://basalt-test/",
			api_key="my-api-key",
			sdk_version="1.0.0",
			sdk_type="py-test"
		)

		api.invoke(mocked_endpoint, { "some": "dto" })

		call_args = mocked_network.fetch.call_args[0]

		self.assertIn("/test-path", call_args[0])

	@parameterized.expand([
		(None),
		({ "tag": "abc" }),
	])
	def test_includes_path_in_url(self, params):
		mocked_network = MagicMock()
		mocked_network.fetch.return_value = (None, { "some": "response" })

		mocked_endpoint = MagicMock()
		mocked_endpoint.prepare_request.return_value = {
			"path": "/test-path",
			"method": "GET",
			"query": params
		}
		mocked_endpoint.decode_response.return_value = (None, { "some": "response" })

		api = Api(
			networker=mocked_network,
			root_url="https://basalt-test/",
			api_key="my-api-key",
			sdk_version="1.0.0",
			sdk_type="py-test"
		)

		api.invoke(mocked_endpoint, { "some": "dto" })

		call_args = mocked_network.fetch.call_args

		self.assertEqual(call_args.kwargs["params"], params)

	def test_passes_headers_to_network(self):
		mocked_network = MagicMock()
		mocked_network.fetch.return_value = (None, { "some": "response" })

		mocked_endpoint = MagicMock()
		mocked_endpoint.prepare_request.return_value = {
			"path": "/test-path",
			"method": "GET",
			"query": {}
		}
		mocked_endpoint.decode_response.return_value = (None, { "some": "response" })

		api = Api(
			networker=mocked_network,
			root_url="https://basalt-test/",
			api_key="my-api-key",
			sdk_version="test-sdk-version",
			sdk_type="test-sdk-type"
		)

		api.invoke(mocked_endpoint, { "some": "dto" })

		headers = mocked_network.fetch.call_args.kwargs["headers"]

		self.assertIn("Authorization", headers)
		self.assertIn("my-api-key", headers["Authorization"])

		self.assertIn("X-BASALT-SDK-VERSION", headers)
		self.assertIn("test-sdk-version", headers["X-BASALT-SDK-VERSION"])

		self.assertIn("X-BASALT-SDK-TYPE", headers)
		self.assertIn("test-sdk-type", headers["X-BASALT-SDK-TYPE"])