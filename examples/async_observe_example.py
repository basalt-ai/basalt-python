"""
Example demonstrating async context manager usage with async_start_observe and async_observe.

This example shows how to use the async versions of start_observe and observe
for proper awaiting in async contexts.
"""

import asyncio
import logging

from basalt.observability import ObserveKind, async_observe, async_start_observe


async def fetch_data(item_id: int) -> dict:
    """Simulate async data fetching."""
    await asyncio.sleep(0.1)  # Simulate I/O delay
    return {"id": item_id, "name": f"Item {item_id}", "status": "active"}


async def process_item(item: dict) -> dict:
    """Simulate async data processing."""
    async with async_observe(name="process_item", kind=ObserveKind.FUNCTION) as span:
        span.set_input(item)

        # Simulate processing
        await asyncio.sleep(0.05)
        result = {
            "id": item["id"],
            "processed": True,
            "result": f"Processed: {item['name']}"
        }

        span.set_output(result)
        return result


async def fetch_and_process(item_id: int) -> dict:
    """Fetch and process an item with nested spans."""
    async with async_observe(name="fetch_and_process", kind=ObserveKind.FUNCTION) as span:
        span.set_input({"item_id": item_id})

        # Fetch data with nested span
        async with async_observe(name="fetch_data", kind=ObserveKind.RETRIEVAL) as fetch_span:
            fetch_span.set_input({"item_id": item_id})
            data = await fetch_data(item_id)
            fetch_span.set_output(data)

        # Process the fetched data
        result = await process_item(data)

        span.set_output(result)
        return result


async def main():
    """Main async workflow demonstrating async_start_observe."""
    logging.basicConfig(level=logging.INFO)

    async with async_start_observe(name="async_workflow_example", feature_slug="async_demo") as root_span:
        logging.info("Starting async workflow...")

        # Set some metadata on the root span
        root_span.set_metadata({"workflow_version": "1.0", "environment": "demo"})

        # Process multiple items concurrently
        item_ids = [1, 2, 3, 4, 5]
        tasks = [fetch_and_process(item_id) for item_id in item_ids]

        results = await asyncio.gather(*tasks)

        # Set the final output
        root_span.set_output({
            "processed_count": len(results),
            "items": results
        })

        logging.info(f"Completed processing {len(results)} items")


if __name__ == "__main__":
    asyncio.run(main())
