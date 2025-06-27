import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from basalt.sdk.monitorsdk import MonitorSDK
from basalt.utils.logger import Logger
from basalt.ressources.monitor.monitorsdk_types import (
    Experiment, Trace, Generation, LogParams, GenerationParams,
    TraceParams, ExperimentParams
)
from basalt.endpoints.monitor.create_experiment import CreateExperimentEndpoint, CreateExperimentEndpointResponse
from basalt.endpoints.monitor.send_trace import CreateTraceEndpoint, CreateTraceEndpointResponse
from basalt.endpoints.monitor.create_generation import CreateGenerationEndpoint, CreateGenerationEndpointResponse
from basalt.endpoints.monitor.create_log import CreateLogEndpoint, CreateLogEndpointResponse

logger = Logger()
mocked_api = MagicMock()
# Make sure async_invoke is an AsyncMock
mocked_api.async_invoke = AsyncMock()

# Mock responses for different endpoints
experiment_response = CreateExperimentEndpointResponse(
    experiment=Experiment(
        id="exp-123",
        feature_slug="test-feature",
        run_id="run-123",
        type="A/B Test",
        name="Test Experiment",
        setup={
            "control_id": "control-123",
            "variation_id": "variation-123"
        }
    )
)

trace_response = CreateTraceEndpointResponse(
    trace=Trace(
        id="trace-123",
        name="Test Trace",
        slug="test-trace",
        metadata={"source": "test"},
        run_id="run-123",
        tool=None,
        model_id=None,
        created_at="2023-01-01T00:00:00Z"
    )
)

generation_response = CreateGenerationEndpointResponse(
    generation=Generation(
        id="gen-123",
        trace_id="trace-123",
        run_id="run-123",
        text="Generated text",
        model_id="gpt-4",
        prompt="Test prompt",
        metadata={"source": "test"},
        created_at="2023-01-01T00:00:00Z"
    )
)

log_response = CreateLogEndpointResponse(
    log={
        "id": "log-123",
        "trace_id": "trace-123",
        "run_id": "run-123",
        "type": "info",
        "message": "Test log message",
        "metadata": {"source": "test"},
        "created_at": "2023-01-01T00:00:00Z"
    }
)


class TestMonitorSDKAsync(unittest.TestCase):
    def setUp(self):
        self.monitor_sdk = MonitorSDK(
            api=mocked_api,
            logger=logger
        )
        # Reset mock calls before each test
        mocked_api.async_invoke.reset_mock()
        
    async def test_async_create_experiment(self):
        """Test asynchronously creating an experiment"""
        # Configure mock
        mocked_api.async_invoke.return_value = (None, experiment_response)
        
        # Call the method
        params = ExperimentParams(
            type="A/B Test",
            name="Test Experiment",
            setup={
                "control_id": "control-123",
                "variation_id": "variation-123"
            }
        )
        result = await self.monitor_sdk.async_create_experiment("test-feature", params)
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result.id, "exp-123")
        self.assertEqual(result.feature_slug, "test-feature")
        self.assertEqual(result.type, "A/B Test")
        
        # Verify correct endpoint was used
        endpoint = mocked_api.async_invoke.call_args[0][0]
        self.assertEqual(endpoint, CreateExperimentEndpoint)
        
        # Verify DTO was created correctly
        dto = mocked_api.async_invoke.call_args[0][1]
        self.assertEqual(dto.feature_slug, "test-feature")
        self.assertEqual(dto.params.type, "A/B Test")
        
    async def test_async_create_trace(self):
        """Test asynchronously creating a trace"""
        # Configure mock
        mocked_api.async_invoke.return_value = (None, trace_response)
        
        # Call the method
        params = TraceParams(
            name="Test Trace",
            metadata={"source": "test"}
        )
        result = await self.monitor_sdk.async_create_trace("test-trace", params)
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result.id, "trace-123")
        self.assertEqual(result.name, "Test Trace")
        self.assertEqual(result.slug, "test-trace")
        
        # Verify correct endpoint was used
        endpoint = mocked_api.async_invoke.call_args[0][0]
        self.assertEqual(endpoint, CreateTraceEndpoint)
        
        # Verify DTO was created correctly
        dto = mocked_api.async_invoke.call_args[0][1]
        self.assertEqual(dto.slug, "test-trace")
        self.assertEqual(dto.params.name, "Test Trace")
        
    async def test_async_create_generation(self):
        """Test asynchronously creating a generation"""
        # Configure mock
        mocked_api.async_invoke.return_value = (None, generation_response)
        
        # Call the method
        params = GenerationParams(
            trace_id="trace-123",
            text="Generated text",
            model_id="gpt-4",
            prompt="Test prompt",
            metadata={"source": "test"}
        )
        result = await self.monitor_sdk.async_create_generation(params)
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result.id, "gen-123")
        self.assertEqual(result.trace_id, "trace-123")
        self.assertEqual(result.text, "Generated text")
        self.assertEqual(result.model_id, "gpt-4")
        
        # Verify correct endpoint was used
        endpoint = mocked_api.async_invoke.call_args[0][0]
        self.assertEqual(endpoint, CreateGenerationEndpoint)
        
        # Verify DTO was created correctly
        dto = mocked_api.async_invoke.call_args[0][1]
        self.assertEqual(dto.params.trace_id, "trace-123")
        self.assertEqual(dto.params.text, "Generated text")
        
    async def test_async_create_log(self):
        """Test asynchronously creating a log"""
        # Configure mock
        mocked_api.async_invoke.return_value = (None, log_response)
        
        # Call the method
        params = LogParams(
            trace_id="trace-123",
            type="info",
            message="Test log message",
            metadata={"source": "test"}
        )
        result = await self.monitor_sdk.async_create_log(params)
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result["id"], "log-123")
        self.assertEqual(result["trace_id"], "trace-123")
        self.assertEqual(result["message"], "Test log message")
        
        # Verify correct endpoint was used
        endpoint = mocked_api.async_invoke.call_args[0][0]
        self.assertEqual(endpoint, CreateLogEndpoint)
        
        # Verify DTO was created correctly
        dto = mocked_api.async_invoke.call_args[0][1]
        self.assertEqual(dto.params.trace_id, "trace-123")
        self.assertEqual(dto.params.message, "Test log message")
        
    async def test_async_error_handling_create_trace(self):
        """Test error handling when asynchronously creating a trace"""
        # Configure mock to return an error
        error = Exception("API Error")
        mocked_api.async_invoke.return_value = (error, None)
        
        # Call the method
        params = TraceParams(name="Test Trace")
        
        with self.assertRaises(Exception) as context:
            await self.monitor_sdk.async_create_trace("non-existent", params)
        
        # Assertions
        self.assertEqual(str(context.exception), "API Error")


def run_async_tests():
    """
    Helper function to run async tests
    """
    loop = asyncio.get_event_loop()
    
    # Create and run the test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMonitorSDKAsync)
    runner = unittest.TextTestRunner()
    
    for test in suite:
        if test._testMethodName.startswith('test_async_'):
            coro = getattr(test, test._testMethodName)()
            loop.run_until_complete(coro)


if __name__ == "__main__":
    run_async_tests()
