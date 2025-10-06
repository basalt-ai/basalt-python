import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from basalt.sdk.datasetsdk import DatasetSDK
from basalt.utils.logger import Logger
from basalt.utils.dtos import DatasetDTO, DatasetRowDTO, ListDatasetsDTO, GetDatasetDTO, CreateDatasetItemDTO
from basalt.endpoints.list_datasets import ListDatasetsEndpoint, ListDatasetsEndpointResponse
from basalt.endpoints.get_dataset import GetDatasetEndpoint, GetDatasetEndpointResponse
from basalt.endpoints.create_dataset_item import CreateDatasetItemEndpoint, CreateDatasetItemEndpointResponse

logger = Logger()
mocked_api = MagicMock()
# Make sure async_invoke is an AsyncMock
mocked_api.async_invoke = AsyncMock()

# Mock responses for different endpoints - same as in test_datasetsdk.py
dataset_list_response = ListDatasetsEndpointResponse(
    datasets=[
        DatasetDTO(
            slug="test-dataset",
            name="Test Dataset",
            columns=["input", "output"]
        ),
        DatasetDTO(
            slug="another-dataset",
            name="Another Dataset",
            columns=["col1", "col2", "col3"]
        )
    ]
)

dataset_get_response = GetDatasetEndpointResponse(
    dataset=DatasetDTO(
        slug="test-dataset",
        name="Test Dataset",
        columns=["input", "output"],
        rows=[
            {
                "values": {
                    "input": "Sample input",
                    "output": "Sample output"
                },
                "name": "Sample Row",
                "idealOutput": "Ideal output",
                "metadata": {"source": "test"}
            }
        ]
    ),
    error=None
)

dataset_add_row_response = CreateDatasetItemEndpointResponse(
    datasetRow=DatasetRowDTO(
        values={"input": "New input", "output": "New output"},
        name="New Row",
        idealOutput="New ideal output",
        metadata={"source": "test"}
    ),
    warning=None,
    error=None
)


class TestDatasetSDKAsync(unittest.TestCase):
    def setUp(self):
        self.dataset_sdk = DatasetSDK(
            api=mocked_api,
            logger=logger
        )
        # Reset mock calls before each test
        mocked_api.async_invoke.reset_mock()

    async def test_async_list_datasets(self):
        """Test asynchronously listing all datasets"""
        # Configure mock
        mocked_api.async_invoke.return_value = (None, dataset_list_response)

        # Call the method
        err, datasets = await self.dataset_sdk.async_list()

        # Assertions
        self.assertIsNone(err)
        self.assertEqual(len(datasets), 2)
        self.assertEqual(datasets[0].slug, "test-dataset")
        self.assertEqual(datasets[0].name, "Test Dataset")
        self.assertEqual(datasets[1].slug, "another-dataset")

        # Verify correct endpoint was used
        endpoint = mocked_api.async_invoke.call_args[0][0]
        self.assertEqual(endpoint, ListDatasetsEndpoint)

    async def test_async_get_dataset(self):
        """Test asynchronously getting a dataset by slug"""
        # Configure mock
        mocked_api.async_invoke.return_value = (None, dataset_get_response)

        # Call the method
        err, dataset = await self.dataset_sdk.async_get("test-dataset")

        # Assertions
        self.assertIsNone(err)
        self.assertEqual(dataset.slug, "test-dataset")
        self.assertEqual(dataset.name, "Test Dataset")
        self.assertEqual(len(dataset.columns), 2)
        self.assertEqual(len(dataset.rows), 1)

        # Verify correct endpoint was used
        endpoint = mocked_api.async_invoke.call_args[0][0]
        self.assertEqual(endpoint, GetDatasetEndpoint)

        # Verify DTO was created correctly
        dto = mocked_api.async_invoke.call_args[0][1]
        self.assertEqual(dto.slug, "test-dataset")

    async def test_async_create_dataset_item(self):
        """Test asynchronously creating a dataset item"""
        # Configure mock
        mocked_api.async_invoke.return_value = (None, dataset_add_row_response)

        # Call the method
        values = {"input": "New input", "output": "New output"}
        err, row, warning = await self.dataset_sdk.async_add_row(
            slug="test-dataset",
            values=values,
            name="New Row",
            ideal_output="New ideal output",
            metadata={"source": "test"}
        )

        # Assertions
        self.assertIsNone(err)
        self.assertIsNone(warning)
        self.assertEqual(row.values, values)
        self.assertEqual(row.name, "New Row")
        self.assertEqual(row.idealOutput, "New ideal output")

        # Verify correct endpoint was used
        endpoint = mocked_api.async_invoke.call_args[0][0]
        self.assertEqual(endpoint, CreateDatasetItemEndpoint)

        # Verify DTO was created correctly
        dto = mocked_api.async_invoke.call_args[0][1]
        self.assertEqual(dto.slug, "test-dataset")
        self.assertEqual(dto.values, values)
        self.assertEqual(dto.name, "New Row")
        self.assertEqual(dto.idealOutput, "New ideal output")

    async def test_async_error_handling_get_dataset(self):
        """Test error handling when asynchronously getting a dataset"""
        # Configure mock to return an error
        error = Exception("API Error")
        mocked_api.async_invoke.return_value = (error, None)

        # Call the method
        err, dataset = await self.dataset_sdk.async_get("non-existent")

        # Assertions
        self.assertIsNotNone(err)
        self.assertIsNone(dataset)
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
        runner.run_test_case(TestDatasetSDKAsync)
    finally:
        runner.close()
