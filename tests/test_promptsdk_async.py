import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from basalt.sdk.promptsdk import PromptSDK
from basalt.utils.logger import Logger
from basalt.utils.dtos import GetPromptDTO, PromptResponse, DescribePromptDTO, DescribePromptResponse, PromptListDTO, PromptListResponse
from basalt.endpoints.get_prompt import GetPromptEndpoint, GetPromptEndpointResponse
from basalt.endpoints.list_prompts import ListPromptsEndpoint, ListPromptsEndpointResponse
from basalt.endpoints.describe_prompt import DescribePromptEndpoint, DescribePromptEndpointResponse

logger = Logger()
mocked_api = MagicMock()
# Make sure async_invoke is an AsyncMock
mocked_api.async_invoke = AsyncMock()

# Mock model for PromptResponse
mock_model = type('Model', (), {
    'provider': 'openai',
    'model': 'gpt-4',
    'version': 'latest',
    'parameters': type('Params', (), {
        'temperature': 0.7,
        'max_length': 100,
        'top_p': 1.0,
        'top_k': None,
        'frequency_penalty': None,
        'presence_penalty': None,
        'response_format': 'text',
        'json_object': None
    })()
})()

# Mock responses for different endpoints
prompt_get_response = GetPromptEndpointResponse(
    warning=None,
    prompt=PromptResponse(
        text="This is a test prompt: {{variable}}",
        slug="test-prompt",
        tag="latest",
        systemText="You are a helpful assistant",
        version="1.0",
        model=mock_model
    )
)

prompt_list_response = ListPromptsEndpointResponse(
    warning=None,
    prompts=[
        PromptListResponse(
            slug="test-prompt-1",
            status="active",
            name="Test Prompt 1",
            description="First test prompt",
            available_versions=["1.0"],
            available_tags=["latest"]
        ),
        PromptListResponse(
            slug="test-prompt-2",
            status="active",
            name="Test Prompt 2",
            description="Second test prompt",
            available_versions=["1.0", "2.0"],
            available_tags=["latest", "stable"]
        )
    ]
)

prompt_describe_response = DescribePromptEndpointResponse(
    warning=None,
    prompt=DescribePromptResponse(
        slug="test-prompt",
        status="active",
        name="Test Prompt",
        description="A test prompt for unit testing",
        available_versions=["1.0", "1.1", "2.0"],
        available_tags=["latest", "stable"],
        variables=[{"name": "variable", "type": "string"}]
    )
)


class TestPromptSDKAsync(unittest.TestCase):
    def setUp(self):
        from basalt.utils.memcache import MemoryCache
        self.prompt_sdk = PromptSDK(
            api=mocked_api,
            cache=MemoryCache(),
            fallback_cache=MemoryCache(),
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
        self.assertIsNotNone(prompt_response)
        self.assertEqual(prompt_response.text, "This is a test prompt: {{variable}}")
        self.assertEqual(prompt_response.version, "1.0")
        self.assertIsNotNone(generation)  # Generation is created for monitoring
        
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
        self.assertIsNotNone(prompts)
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
        self.assertIsNotNone(prompts)
        self.assertEqual(len(prompts), 2)
        
        # Verify DTO was created correctly
        dto = mocked_api.async_invoke.call_args[0][1]
        self.assertEqual(dto.featureSlug, "test-feature")
        
    async def test_async_describe_prompt(self):
        """Test asynchronously describing a prompt"""
        # Configure mock
        mocked_api.async_invoke.return_value = (None, prompt_describe_response)
        
        # Call the method
        err, prompt_description = await self.prompt_sdk.async_describe("test-prompt")
        
        # Assertions
        self.assertIsNone(err)
        self.assertIsNotNone(prompt_description)
        self.assertEqual(prompt_description.slug, "test-prompt")
        self.assertEqual(prompt_description.name, "Test Prompt")
        self.assertEqual(len(prompt_description.available_versions), 3)
        self.assertIn("1.0", prompt_description.available_versions)
        
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
        self.assertIsNotNone(prompt_response)
        # Variables should be replaced in the response
        self.assertEqual(prompt_response.text, "This is a test prompt: test-value")
        
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


class AsyncTestRunner:
    """Helper class to run async tests properly"""
    
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def run_test_case(self, test_case_class):
        """Run all async test methods in a test case"""
        suite = unittest.TestLoader().loadTestsFromTestCase(test_case_class)
        
        for test in suite:
            test_method = getattr(test, test._testMethodName)
            if asyncio.iscoroutinefunction(test_method):
                try:
                    test.setUp()
                    self.loop.run_until_complete(test_method())
                    print(f"✓ {test._testMethodName}")
                except Exception as e:
                    print(f"✗ {test._testMethodName}: {e}")
            else:
                # Run sync tests normally
                try:
                    test.setUp()
                    test_method()
                    print(f"✓ {test._testMethodName}")
                except Exception as e:
                    print(f"✗ {test._testMethodName}: {e}")
    
    def close(self):
        self.loop.close()


if __name__ == "__main__":
    runner = AsyncTestRunner()
    try:
        runner.run_test_case(TestPromptSDKAsync)
    finally:
        runner.close()