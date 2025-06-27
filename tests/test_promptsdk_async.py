import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

from basalt.sdk.promptsdk import PromptSDK
from basalt.utils.logger import Logger
from basalt.utils.dtos import PromptDTO, PromptVersionDTO, PromptVariablesDTO, GetPromptDTO
from basalt.endpoints.get_prompt import GetPromptEndpoint, GetPromptEndpointResponse
from basalt.endpoints.list_prompts import ListPromptsEndpoint, ListPromptsEndpointResponse
from basalt.endpoints.describe_prompt import DescribePromptEndpoint, DescribePromptEndpointResponse

logger = Logger()
mocked_api = MagicMock()
# Make sure async_invoke is an AsyncMock
mocked_api.async_invoke = AsyncMock()

# Mock responses for different endpoints
prompt_get_response = GetPromptEndpointResponse(
    prompt=PromptDTO(
        slug="test-prompt",
        feature_slug="test-feature",
        name="Test Prompt",
        description="A test prompt",
        text="This is a test prompt with {{variable}}",
        version="1.0",
        tags=["test"],
        variables=[PromptVariablesDTO(name="variable", description="A test variable")]
    )
)

prompt_list_response = ListPromptsEndpointResponse(
    prompts=[
        PromptDTO(
            slug="test-prompt-1",
            feature_slug="test-feature",
            name="Test Prompt 1",
            version="1.0"
        ),
        PromptDTO(
            slug="test-prompt-2",
            feature_slug="test-feature",
            name="Test Prompt 2",
            version="1.0"
        )
    ]
)

prompt_describe_response = DescribePromptEndpointResponse(
    prompt=PromptDTO(
        slug="test-prompt",
        feature_slug="test-feature",
        name="Test Prompt",
        description="A test prompt",
        text="This is a test prompt with {{variable}}",
        version="1.0",
        tags=["test"],
        variables=[PromptVariablesDTO(name="variable", description="A test variable")]
    ),
    versions=[
        PromptVersionDTO(
            version="1.0",
            text="This is a test prompt with {{variable}}",
            created_at="2023-01-01T00:00:00Z"
        ),
        PromptVersionDTO(
            version="0.9",
            text="This is an older test prompt with {{variable}}",
            created_at="2022-12-01T00:00:00Z"
        )
    ]
)


class TestPromptSDKAsync(unittest.TestCase):
    def setUp(self):
        self.prompt_sdk = PromptSDK(
            api=mocked_api,
            logger=logger
        )
        # Reset mock calls before each test
        mocked_api.async_invoke.reset_mock()
        
    async def test_async_get_prompt(self):
        """Test asynchronously getting a prompt"""
        # Configure mock
        mocked_api.async_invoke.return_value = (None, prompt_get_response)
        
        # Call the method
        err, prompt_response, generation = await self.prompt_sdk.async_get("test-prompt")
        
        # Assertions
        self.assertIsNone(err)
        self.assertEqual(prompt_response.text, "This is a test prompt with {{variable}}")
        self.assertEqual(prompt_response.slug, "test-prompt")
        self.assertIsNone(generation)  # No monitoring in this test
        
        # Verify correct endpoint was used
        endpoint = mocked_api.async_invoke.call_args[0][0]
        self.assertEqual(endpoint, GetPromptEndpoint)
        
        # Verify DTO was created correctly
        dto = mocked_api.async_invoke.call_args[0][1]
        self.assertEqual(dto.slug, "test-prompt")
        
    async def test_async_list_prompts(self):
        """Test asynchronously listing prompts"""
        # Configure mock
        mocked_api.async_invoke.return_value = (None, prompt_list_response)
        
        # Call the method
        err, prompts = await self.prompt_sdk.async_list()
        
        # Assertions
        self.assertIsNone(err)
        self.assertEqual(len(prompts), 2)
        self.assertEqual(prompts[0].slug, "test-prompt-1")
        self.assertEqual(prompts[1].slug, "test-prompt-2")
        
        # Verify correct endpoint was used
        endpoint = mocked_api.async_invoke.call_args[0][0]
        self.assertEqual(endpoint, ListPromptsEndpoint)
        
    async def test_async_list_prompts_with_feature_filter(self):
        """Test asynchronously listing prompts with feature filter"""
        # Configure mock
        mocked_api.async_invoke.return_value = (None, prompt_list_response)
        
        # Call the method
        err, prompts = await self.prompt_sdk.async_list(feature_slug="test-feature")
        
        # Assertions
        self.assertIsNone(err)
        self.assertEqual(len(prompts), 2)
        
        # Verify DTO was created correctly
        dto = mocked_api.async_invoke.call_args[0][1]
        self.assertEqual(dto.feature_slug, "test-feature")
        
    async def test_async_describe_prompt(self):
        """Test asynchronously describing a prompt"""
        # Configure mock
        mocked_api.async_invoke.return_value = (None, prompt_describe_response)
        
        # Call the method
        err, prompt_description = await self.prompt_sdk.async_describe("test-prompt")
        
        # Assertions
        self.assertIsNone(err)
        self.assertEqual(prompt_description.prompt.slug, "test-prompt")
        self.assertEqual(len(prompt_description.versions), 2)
        self.assertEqual(prompt_description.versions[0].version, "1.0")
        self.assertEqual(prompt_description.versions[1].version, "0.9")
        
        # Verify correct endpoint was used
        endpoint = mocked_api.async_invoke.call_args[0][0]
        self.assertEqual(endpoint, DescribePromptEndpoint)
        
    async def test_async_get_prompt_with_variables(self):
        """Test asynchronously getting a prompt with variables replaced"""
        # Configure mock
        mocked_api.async_invoke.return_value = (None, prompt_get_response)
        
        # Call the method
        err, prompt_response, generation = await self.prompt_sdk.async_get(
            "test-prompt",
            variables={"variable": "test-value"}
        )
        
        # Assertions
        self.assertIsNone(err)
        # Variables should be replaced in the response
        self.assertEqual(prompt_response.text, "This is a test prompt with test-value")
        
    async def test_async_get_prompt_error_handling(self):
        """Test error handling when asynchronously getting a prompt"""
        # Configure mock to return an error
        error = Exception("API Error")
        mocked_api.async_invoke.return_value = (error, None)
        
        # Call the method
        err, prompt_response, generation = await self.prompt_sdk.async_get("non-existent")
        
        # Assertions
        self.assertIsNotNone(err)
        self.assertIsNone(prompt_response)
        self.assertIsNone(generation)
        self.assertEqual(str(err), "API Error")


def run_async_tests():
    """
    Helper function to run async tests
    """
    loop = asyncio.get_event_loop()
    
    # Create and run the test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPromptSDKAsync)
    runner = unittest.TextTestRunner()
    
    for test in suite:
        if test._testMethodName.startswith('test_async_'):
            coro = getattr(test, test._testMethodName)()
            loop.run_until_complete(coro)


if __name__ == "__main__":
    run_async_tests()
