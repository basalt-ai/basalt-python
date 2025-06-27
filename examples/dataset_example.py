"""
Dataset Example - Demonstrates how to use the Basalt Dataset SDK

This example shows how to:
1. List all available datasets
2. Get a specific dataset with its rows
3. Create a new dataset item
4. Work with dataset objects

Make sure to set your API key as an environment variable:
export BASALT_API_KEY=your_api_key
"""

import os
import sys
from typing import Dict, List, Optional

from basalt import Basalt

# Set up the Basalt client
api_key = os.getenv("BASALT_API_KEY")
if not api_key:
    print("Please set your BASALT_API_KEY environment variable")
    sys.exit(1)

basalt = Basalt(api_key=api_key)

def list_datasets():
    """List all datasets in the workspace"""
    print("\n=== Listing all datasets ===")
    err, datasets = basalt.datasets.list()
    
    if err:
        print(f"Error listing datasets: {err}")
        return None
    
    print(f"Found {len(datasets)} datasets:")
    for dataset in datasets:
        print(f"- {dataset.name} (slug: {dataset.slug})")
        print(f"  Columns: {', '.join(dataset.columns)}")
    
    # Return the first dataset slug for other examples to use
    return datasets[0].slug if datasets else None

def get_dataset(slug: str):
    """Get a specific dataset by slug"""
    print(f"\n=== Getting dataset: {slug} ===")
    err, dataset = basalt.datasets.get(slug)
    
    if err:
        print(f"Error getting dataset: {err}")
        return
    
    print(f"Dataset: {dataset.name}")
    print(f"Columns: {', '.join(dataset.columns)}")
    
    if dataset.rows:
        print(f"Number of rows: {len(dataset.rows)}")
        print("First few rows:")
        for i, row in enumerate(dataset.rows[:3]):  # Show up to 3 rows
            print(f"Row {i+1}:")
            print(f"  Values: {row.values}")
            if row.idealOutput:
                print(f"  Ideal output: {row.idealOutput}")
            if row.metadata:
                print(f"  Metadata: {row.metadata}")
    else:
        print("Dataset has no rows")

def add_row_to_dataset(slug: str):
    """Create a new dataset item"""
    print(f"\n=== Creating new dataset item in {slug} ===")
    
    # Get the dataset to understand its structure
    err, dataset = basalt.datasets.get(slug)
    if err or not dataset:
        print(f"Error getting dataset structure: {err}")
        return
    
    # Create values for all columns
    values = {}
    for column in dataset.columns:
        values[column] = f"Example value for {column}"
    
    # Create the item
    err, row, warning = basalt.datasets.addRow(
        slug=slug,
        values=values,
        name="Example Row",
        ideal_output="Example ideal output",
        metadata={"source": "Python SDK example", "timestamp": "2025-06-27"}
    )
    
    if err:
        print(f"Error creating dataset item: {err}")
        return
    
    if warning:
        print(f"Warning: {warning}")
    
    print("Dataset item created successfully:")
    print(f"Values: {row.values}")
    print(f"Name: {row.name}")
    print(f"Ideal output: {row.idealOutput}")
    print(f"Metadata: {row.metadata}")

def work_with_dataset_objects(slug: str):
    """Demonstrate working with Dataset objects"""
    print(f"\n=== Working with dataset objects for {slug} ===")
    
    dataset = basalt.datasets.get_dataset_object(slug)
    if not dataset:
        print("Failed to get dataset object")
        return
        
    print(f"Dataset object: {dataset.name} with {len(dataset.columns)} columns")
    
    # Add a new row to the dataset object
    values = {column: f"New object value for {column}" for column in dataset.columns}
    
    row = basalt.datasets.add_row(
        dataset=dataset,
        values=values,
        name="Object Example Row",
        ideal_output="Object example ideal output",
        metadata={"created_by": "dataset_object_example"}
    )
    
    if row:
        print("Added new row to dataset object:")
        print(f"Values: {row.values}")
        print(f"Name: {row.name}")
        print(f"Dataset now has {len(dataset.rows)} rows")
    else:
        print("Failed to add row to dataset object")

def main():
    """Main function to run the example"""
    print("Basalt Dataset Example")
    
    # List all datasets and get the first one's slug
    dataset_slug = list_datasets()
    if not dataset_slug:
        print("No datasets available to continue the example")
        return
    
    # Get details for the selected dataset
    get_dataset(dataset_slug)
    
    # Create a new item in the dataset
    create_dataset_item(dataset_slug)
    
    # Work with dataset objects
    work_with_dataset_objects(dataset_slug)

if __name__ == "__main__":
    main()
