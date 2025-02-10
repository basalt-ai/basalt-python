import unittest
from basalt.endpoints.get_prompt import GetPromptEndpoint
from basalt.utils.dtos import GetPromptDTO


class TestGetPromptEndpoint(unittest.TestCase):
	
	def test_includes_slug_in_path(self):
		result = GetPromptEndpoint.prepare_request(GetPromptDTO(slug="my-complex-slug-that-should-be-unique"))

		self.assertIn("my-complex-slug-that-should-be-unique", result["path"])

	def test_includes_tags_as_queryparam(self):
		result = GetPromptEndpoint.prepare_request(GetPromptDTO(slug="slug", tag="abc"))

		self.assertEqual(result["query"].get("tag"), "abc")

	def test_includes_version_as_queryparam(self):
		result = GetPromptEndpoint.prepare_request(GetPromptDTO(slug="slug", version="2.0"))

		self.assertEqual(result["query"].get("version"), "2.0")

	def test_decodes_valid_response(self):
		response = {
			"warning": "This is a warning",
			"prompt": {
				"text": "Valid prompt text",
				"model": {
					"provider": "open-ai",
            		"model": "gpt-4o",
					"version": "latest",
					"parameters": {
						"temperature": 0.7,
						"topP": 1,
						"maxLength": 4096,
						"responseFormat": "text"
					}
				}
			}
		}

		exception, decoded = GetPromptEndpoint.decode_response(response)

		self.assertIsNone(exception)
		self.assertEqual(decoded.warning, "This is a warning")
		self.assertEqual(decoded.prompt.text, "Valid prompt text")
		self.assertEqual(decoded.prompt.model.model, "gpt-4o")
		self.assertEqual(decoded.prompt.model.provider, "open-ai")
