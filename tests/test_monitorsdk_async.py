import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from basalt.sdk.monitorsdk import MonitorSDK
from basalt.utils.logger import Logger
from basalt.ressources.monitor.experiment_types import ExperimentParams
from basalt.ressources.monitor.trace_types import TraceParams
from basalt.ressources.monitor.generation_types import GenerationParams
from basalt.ressources.monitor.log_types import LogParams
from basalt.objects.experiment import Experiment
from basalt.objects.trace import Trace
from basalt.objects.generation import Generation
from basalt.objects.log import Log
from basalt.endpoints.monitor.create_experiment import CreateExperimentEndpoint, Output as ExperimentOutput

logger = Logger()
mocked_api = MagicMock()
# Make sure async_invoke is an AsyncMock
mocked_api.async_invoke = AsyncMock()

# Mock experiment data that matches the actual Experiment structure
experiment_data = {
    "id": "exp-123",
    "featureSlug": "test-feature",
    "name": "Test Experiment",
    "createdAt": "2023-01-01T00:00:00Z"
}

experiment_output = ExperimentOutput(
    experiment=type('Experiment', (), experiment_data)()
)

# Set experiment attributes properly
experiment_output.experiment.id = experiment_data["id"]
experiment_output.experiment.feature_slug = experiment_data["featureSlug"]
experiment_output.experiment.name = experiment_data["name"]
experiment_output.experiment.created_at = experiment_data["createdAt"]


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
        mocked_api.async_invoke.return_value = (None, experiment_output)
        
        # Call the method
        params = {"name": "Test Experiment"}
        err, result = await self.monitor_sdk.async_create_experiment("test-feature", params)
        
        # Assertions
        self.assertIsNone(err)
        self.assertIsNotNone(result)
        self.assertEqual(result.id, "exp-123")
        self.assertEqual(result.feature_slug, "test-feature")
        self.assertEqual(result.name, "Test Experiment")
        
        # Verify correct endpoint was used
        endpoint = mocked_api.async_invoke.call_args[0][0]
        self.assertEqual(endpoint, CreateExperimentEndpoint)
        
        # Verify DTO was created correctly
        dto = mocked_api.async_invoke.call_args[0][1]
        self.assertEqual(dto.feature_slug, "test-feature")
        self.assertEqual(dto.name, "Test Experiment")
        
    async def test_async_create_trace(self):
        """Test asynchronously creating a trace"""
        # Call the method - traces are created directly without API calls
        params = {
            "name": "Test Trace",
            "metadata": {"source": "test"}
        }
        result = await self.monitor_sdk.async_create_trace("test-trace", params)
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertIsInstance(result, Trace)
        self.assertEqual(result.name, "Test Trace")
        self.assertEqual(result.feature_slug, "test-trace")
        self.assertEqual(result.metadata, {"source": "test"})
        
    async def test_async_create_generation(self):
        """Test asynchronously creating a generation"""
        # Call the method - generations are created directly without API calls
        params = {
            "name": "Test Generation",
            "input": "Test input",
            "metadata": {"source": "test"}
        }
        result = await self.monitor_sdk.async_create_generation(params)
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertIsInstance(result, Generation)
        self.assertEqual(result.input, "Test input")
        self.assertEqual(result.name, "Test Generation")
        self.assertEqual(result.metadata, {"source": "test"})
        # Options will be None since not provided
        self.assertIsNone(result.options)
        
    async def test_async_create_log(self):
        """Test asynchronously creating a log"""
        # Call the method - logs are created directly without API calls
        params = {
            "name": "Test Log",
            "input": "Test log input",
            "metadata": {"source": "test"}
        }
        result = await self.monitor_sdk.async_create_log(params)
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertIsInstance(result, Log)
        self.assertEqual(result.name, "Test Log")
        self.assertEqual(result.input, "Test log input")
        self.assertEqual(result.metadata, {"source": "test"})
        
    async def test_async_error_handling_create_experiment(self):
        """Test error handling when asynchronously creating an experiment"""
        # Configure mock to return an error
        error = Exception("API Error")
        mocked_api.async_invoke.return_value = (error, None)
        
        # Call the method
        params = {"name": "Test Experiment"}
        
        err, result = await self.monitor_sdk.async_create_experiment("test-feature", params)
        
        # Assertions
        self.assertIsNotNone(err)
        self.assertIsNone(result)
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
        runner.run_test_case(TestMonitorSDKAsync)
    finally:
        runner.close()
