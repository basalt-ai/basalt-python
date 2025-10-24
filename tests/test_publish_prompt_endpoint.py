import unittest
from basalt.endpoints.publish_prompt import PublishPromptEndpoint
from basalt.utils.dtos import PublishPromptDTO


class TestPublishPromptEndpoint(unittest.TestCase):
	"""Test suite for the PublishPromptEndpoint"""

	def test_includes_slug_in_path(self):
		"""Test that the slug is included in the request path"""
		result = PublishPromptEndpoint.prepare_request(
			PublishPromptDTO(slug="my-unique-prompt-slug", new_tag="production", version="1.0.0")
		)

		self.assertIn("my-unique-prompt-slug", result["path"])
		self.assertEqual(result["path"], "/prompts/my-unique-prompt-slug/publish")

	def test_uses_post_method(self):
		"""Test that the endpoint uses POST method"""
		result = PublishPromptEndpoint.prepare_request(
			PublishPromptDTO(slug="test-slug", new_tag="prod", version="1.0.0")
		)

		self.assertEqual(result["method"], "POST")

	def test_includes_new_tag_in_body(self):
		"""Test that newTag is included in the request body"""
		result = PublishPromptEndpoint.prepare_request(
			PublishPromptDTO(slug="slug", new_tag="production", version="1.0.0")
		)

		self.assertEqual(result["body"]["newTag"], "production")

	def test_includes_version_in_body_when_provided(self):
		"""Test that version is included in body when provided"""
		result = PublishPromptEndpoint.prepare_request(
			PublishPromptDTO(slug="slug", new_tag="prod", version="2.5.1")
		)

		self.assertEqual(result["body"]["version"], "2.5.1")

	def test_includes_tag_in_body_when_provided(self):
		"""Test that tag is included in body when provided"""
		result = PublishPromptEndpoint.prepare_request(
			PublishPromptDTO(slug="slug", new_tag="prod", tag="latest")
		)

		self.assertEqual(result["body"]["tag"], "latest")

	def test_omits_version_when_not_provided(self):
		"""Test that version is omitted from body when not provided"""
		result = PublishPromptEndpoint.prepare_request(
			PublishPromptDTO(slug="slug", new_tag="prod", tag="latest")
		)

		self.assertNotIn("version", result["body"])

	def test_omits_tag_when_not_provided(self):
		"""Test that tag is omitted from body when not provided"""
		result = PublishPromptEndpoint.prepare_request(
			PublishPromptDTO(slug="slug", new_tag="prod", version="1.0.0")
		)

		self.assertNotIn("tag", result["body"])

	def test_decodes_valid_response(self):
		"""Test that a valid response is properly decoded"""
		response = {
			"deploymentTag": {
				"id": "test-deployment-tag-id-123",
				"label": "production"
			}
		}

		exception, decoded = PublishPromptEndpoint.decode_response(response)

		self.assertIsNone(exception)
		self.assertIsNotNone(decoded)
		self.assertEqual(decoded.deploymentTag.id, "test-deployment-tag-id-123")
		self.assertEqual(decoded.deploymentTag.label, "production")
		self.assertIsNone(decoded.error)

	def test_decodes_error_response(self):
		"""Test that an error response is properly decoded"""
		response = {
			"error": "Prompt not found"
		}

		exception, decoded = PublishPromptEndpoint.decode_response(response)

		self.assertIsNone(exception)
		self.assertIsNotNone(decoded)
		self.assertEqual(decoded.error, "Prompt not found")
		self.assertIsNone(decoded.deploymentTag)

	def test_handles_malformed_response(self):
		"""Test that a malformed response returns an exception"""
		response = {
			"unexpected": "data"
		}

		exception, decoded = PublishPromptEndpoint.decode_response(response)

		self.assertIsNotNone(exception)
		self.assertIsNone(decoded)

	def test_handles_both_version_and_tag(self):
		"""Test that both version and tag can be provided"""
		result = PublishPromptEndpoint.prepare_request(
			PublishPromptDTO(slug="slug", new_tag="prod", version="1.0.0", tag="latest")
		)

		self.assertEqual(result["body"]["version"], "1.0.0")
		self.assertEqual(result["body"]["tag"], "latest")
		self.assertEqual(result["body"]["newTag"], "prod")


if __name__ == '__main__':
	unittest.main()
