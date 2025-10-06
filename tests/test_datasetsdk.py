import unittest
from unittest.mock import MagicMock
from parameterized import parameterized

from basalt.sdk.datasetsdk import DatasetSDK
from basalt.utils.logger import Logger
from basalt.utils.dtos import DatasetDTO, DatasetRowDTO
from basalt.endpoints.list_datasets import ListDatasetsEndpoint, ListDatasetsEndpointResponse
from basalt.endpoints.get_dataset import GetDatasetEndpoint, GetDatasetEndpointResponse
from basalt.endpoints.create_dataset_item import CreateDatasetItemEndpoint, CreateDatasetItemEndpointResponse

logger = Logger()
mocked_api = MagicMock()

# Mock responses for different endpoints
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


class TestDatasetSDK(unittest.TestCase):
    def setUp(self):
        self.dataset_sdk = DatasetSDK(
            api=mocked_api,
            logger=logger
        )
        
    def test_list_datasets(self):
        """Test listing all datasets"""
        # Configure mock
        mocked_api.invoke_sync.return_value = (None, dataset_list_response)
        
        # Call the method
        err, datasets = self.dataset_sdk.list_sync()
        
        # Assertions
        self.assertIsNone(err)
        self.assertEqual(len(datasets), 2)
        self.assertEqual(datasets[0].slug, "test-dataset")
        self.assertEqual(datasets[0].name, "Test Dataset")
        self.assertEqual(datasets[1].slug, "another-dataset")
        
        # Verify correct endpoint was used
        endpoint = mocked_api.invoke_sync.call_args[0][0]
        self.assertEqual(endpoint, ListDatasetsEndpoint)
        
    def test_get_dataset(self):
        """Test getting a dataset by slug"""
        # Configure mock
        mocked_api.invoke_sync.return_value = (None, dataset_get_response)
        
        # Call the method
        err, dataset = self.dataset_sdk.get_sync("test-dataset")
        
        # Assertions
        self.assertIsNone(err)
        self.assertEqual(dataset.slug, "test-dataset")
        self.assertEqual(dataset.name, "Test Dataset")
        self.assertEqual(len(dataset.columns), 2)
        self.assertEqual(len(dataset.rows), 1)
        
        # Verify correct endpoint was used
        endpoint = mocked_api.invoke_sync.call_args[0][0]
        self.assertEqual(endpoint, GetDatasetEndpoint)
        
        # Verify DTO was created correctly
        dto = mocked_api.invoke_sync.call_args[0][1]
        self.assertEqual(dto.slug, "test-dataset")
        
    def test_create_dataset_item(self):
        """Test creating a dataset item"""
        # Configure mock
        mocked_api.invoke_sync.return_value = (None, dataset_add_row_response)
        
        # Call the method
        values = {"input": "New input", "output": "New output"}
        err, row, warning = self.dataset_sdk.add_row_sync(
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
        endpoint = mocked_api.invoke_sync.call_args[0][0]
        self.assertEqual(endpoint, CreateDatasetItemEndpoint)
        
        # Verify DTO was created correctly
        dto = mocked_api.invoke_sync.call_args[0][1]
        self.assertEqual(dto.slug, "test-dataset")
        self.assertEqual(dto.values, values)
        self.assertEqual(dto.name, "New Row")
        self.assertEqual(dto.idealOutput, "New ideal output")
        
    def test_error_handling_get_dataset(self):
        """Test error handling when getting a dataset"""
        # Configure mock to return an error
        mocked_api.invoke_sync.return_value = (Exception("API Error"), None)
        
        # Call the method
        err, dataset = self.dataset_sdk.get_sync("non-existent")
        
        # Assertions
        self.assertIsNotNone(err)
        self.assertIsNone(dataset)
        self.assertEqual(str(err), "API Error")


if __name__ == "__main__":
    unittest.main()
