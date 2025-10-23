import unittest
from unittest.mock import MagicMock, AsyncMock
from parameterized import parameterized

from basalt.sdk.promptsdk import PromptSDK
from basalt.utils.logger import Logger
from basalt.utils.dtos import PublishPromptDTO, DeploymentTagResponse
from basalt.endpoints.publish_prompt import PublishPromptEndpoint, PublishPromptEndpointResponse

logger = Logger()


class TestPublishPromptSync(unittest.TestCase):
	"""Test suite for the publish_sync method"""

	def setUp(self):
		"""Set up test fixtures"""
		self.mocked_api = MagicMock()
		self.mocked_cache = MagicMock()
		self.mocked_cache.get.return_value = None
		self.fallback_cache = MagicMock()
		self.fallback_cache.get.return_value = None

		# Mock successful response
		self.mocked_api.invoke_sync.return_value = (None, PublishPromptEndpointResponse(
			deploymentTag=DeploymentTagResponse(
				id="test-tag-id-123",
				label="production"
			),
			error=None
		))

		self.prompt_sdk = PromptSDK(
			self.mocked_api,
			cache=self.mocked_cache,
			fallback_cache=self.fallback_cache,
			logger=logger
		)

	def test_uses_correct_endpoint(self):
		"""Test that publish_sync uses the PublishPromptEndpoint"""
		self.prompt_sdk.publish_sync("test-slug", "production", version="1.0.0")

		endpoint = self.mocked_api.invoke_sync.call_args[0][0]
		self.assertEqual(endpoint, PublishPromptEndpoint)

	@parameterized.expand([
		# (slug, new_tag, version, tag)
		("my-prompt", "production", "1.0.0", None),
		("my-prompt", "staging", None, "latest"),
		("another-prompt", "prod", "2.5.1", None),
		("test-slug", "custom-tag", None, "development"),
	])
	def test_passes_correct_dto(self, slug, new_tag, version, tag):
		"""Test that the correct DTO is passed to the API"""
		self.prompt_sdk.publish_sync(slug, new_tag, version=version, tag=tag)

		dto = self.mocked_api.invoke_sync.call_args[0][1]

		self.assertEqual(dto, PublishPromptDTO(
			slug=slug,
			new_tag=new_tag,
			version=version,
			tag=tag
		))

	def test_returns_success_response(self):
		"""Test that a successful response is properly returned"""
		err, deployment_tag = self.prompt_sdk.publish_sync(
			"test-slug",
			"production",
			version="1.0.0"
		)

		self.assertIsNone(err)
		self.assertIsNotNone(deployment_tag)
		self.assertEqual(deployment_tag.id, "test-tag-id-123")
		self.assertEqual(deployment_tag.label, "production")

	def test_forwards_api_error(self):
		"""Test that API errors are properly forwarded"""
		error_message = "Prompt not found"
		self.mocked_api.invoke_sync.return_value = (Exception(error_message), None)

		err, deployment_tag = self.prompt_sdk.publish_sync(
			"test-slug",
			"production",
			version="1.0.0"
		)

		self.assertIsInstance(err, Exception)
		self.assertEqual(str(err), error_message)
		self.assertIsNone(deployment_tag)

	def test_validates_version_or_tag_required(self):
		"""Test that either version or tag must be provided"""
		err, deployment_tag = self.prompt_sdk.publish_sync(
			"test-slug",
			"production"
			# Neither version nor tag provided
		)

		self.assertIsInstance(err, ValueError)
		self.assertEqual(str(err), "Either version or tag must be provided")
		self.assertIsNone(deployment_tag)

	def test_accepts_version_only(self):
		"""Test that providing only version works"""
		err, deployment_tag = self.prompt_sdk.publish_sync(
			"test-slug",
			"production",
			version="1.0.0"
		)

		self.assertIsNone(err)
		self.assertIsNotNone(deployment_tag)

	def test_accepts_tag_only(self):
		"""Test that providing only tag works"""
		err, deployment_tag = self.prompt_sdk.publish_sync(
			"test-slug",
			"production",
			tag="latest"
		)

		self.assertIsNone(err)
		self.assertIsNotNone(deployment_tag)

	def test_handles_both_version_and_tag(self):
		"""Test that providing both version and tag works"""
		err, deployment_tag = self.prompt_sdk.publish_sync(
			"test-slug",
			"production",
			version="1.0.0",
			tag="latest"
		)

		self.assertIsNone(err)
		self.assertIsNotNone(deployment_tag)


class TestPublishPromptAsync(unittest.TestCase):
	"""Test suite for the async publish method"""

	def setUp(self):
		"""Set up test fixtures"""
		self.mocked_api = MagicMock()
		self.mocked_api.invoke = AsyncMock()
		self.mocked_cache = MagicMock()
		self.mocked_cache.get.return_value = None
		self.fallback_cache = MagicMock()
		self.fallback_cache.get.return_value = None

		# Mock successful response
		self.mocked_api.invoke.return_value = (None, PublishPromptEndpointResponse(
			deploymentTag=DeploymentTagResponse(
				id="async-tag-id-456",
				label="staging"
			),
			error=None
		))

		self.prompt_sdk = PromptSDK(
			self.mocked_api,
			cache=self.mocked_cache,
			fallback_cache=self.fallback_cache,
			logger=logger
		)

	async def test_async_uses_correct_endpoint(self):
		"""Test that async publish uses the PublishPromptEndpoint"""
		await self.prompt_sdk.publish("test-slug", "staging", version="2.0.0")

		endpoint = self.mocked_api.invoke.call_args[0][0]
		self.assertEqual(endpoint, PublishPromptEndpoint)

	async def test_async_passes_correct_dto(self):
		"""Test that the correct DTO is passed to the API in async mode"""
		slug = "my-async-prompt"
		new_tag = "production"
		version = "3.0.0"

		await self.prompt_sdk.publish(slug, new_tag, version=version)

		dto = self.mocked_api.invoke.call_args[0][1]

		self.assertEqual(dto, PublishPromptDTO(
			slug=slug,
			new_tag=new_tag,
			version=version,
			tag=None
		))

	async def test_async_returns_success_response(self):
		"""Test that a successful async response is properly returned"""
		err, deployment_tag = await self.prompt_sdk.publish(
			"test-slug",
			"staging",
			version="2.0.0"
		)

		self.assertIsNone(err)
		self.assertIsNotNone(deployment_tag)
		self.assertEqual(deployment_tag.id, "async-tag-id-456")
		self.assertEqual(deployment_tag.label, "staging")

	async def test_async_forwards_api_error(self):
		"""Test that async API errors are properly forwarded"""
		error_message = "Version not found"
		self.mocked_api.invoke.return_value = (Exception(error_message), None)

		err, deployment_tag = await self.prompt_sdk.publish(
			"test-slug",
			"staging",
			version="2.0.0"
		)

		self.assertIsInstance(err, Exception)
		self.assertEqual(str(err), error_message)
		self.assertIsNone(deployment_tag)

	async def test_async_validates_version_or_tag_required(self):
		"""Test that async method validates version or tag requirement"""
		err, deployment_tag = await self.prompt_sdk.publish(
			"test-slug",
			"production"
			# Neither version nor tag provided
		)

		self.assertIsInstance(err, ValueError)
		self.assertEqual(str(err), "Either version or tag must be provided")
		self.assertIsNone(deployment_tag)

	async def test_async_accepts_tag_only(self):
		"""Test that async method accepts tag only"""
		err, deployment_tag = await self.prompt_sdk.publish(
			"test-slug",
			"production",
			tag="latest"
		)

		self.assertIsNone(err)
		self.assertIsNotNone(deployment_tag)


# Helper to run async tests
def run_async_test(coro):
	"""Helper function to run async tests"""
	import asyncio
	loop = asyncio.get_event_loop()
	return loop.run_until_complete(coro)


# Add async test wrappers
def test_async_uses_correct_endpoint():
	test_case = TestPublishPromptAsync()
	test_case.setUp()
	run_async_test(test_case.test_async_uses_correct_endpoint())


def test_async_passes_correct_dto():
	test_case = TestPublishPromptAsync()
	test_case.setUp()
	run_async_test(test_case.test_async_passes_correct_dto())


def test_async_returns_success_response():
	test_case = TestPublishPromptAsync()
	test_case.setUp()
	run_async_test(test_case.test_async_returns_success_response())


def test_async_forwards_api_error():
	test_case = TestPublishPromptAsync()
	test_case.setUp()
	run_async_test(test_case.test_async_forwards_api_error())


def test_async_validates_version_or_tag_required():
	test_case = TestPublishPromptAsync()
	test_case.setUp()
	run_async_test(test_case.test_async_validates_version_or_tag_required())


def test_async_accepts_tag_only():
	test_case = TestPublishPromptAsync()
	test_case.setUp()
	run_async_test(test_case.test_async_accepts_tag_only())


if __name__ == '__main__':
	unittest.main()
