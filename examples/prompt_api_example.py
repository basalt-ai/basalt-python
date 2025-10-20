"""Basalt Prompt SDK Example (New Exception-Based API).

This script demonstrates how to use the new exception-based Basalt Prompt SDK.
It showcases both synchronous and asynchronous operations with proper error handling.

Before running this script:
1. Set your API key: export BASALT_API_KEY="your-api-key"
2. Set a test prompt slug: export BASALT_TEST_PROMPT_SLUG="your-prompt-slug"
3. Install dependencies: pip install basalt-sdk

Reference: See MIGRATION_GUIDE.md for details on the new exception-based API.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

from basalt._internal.exceptions import (
    BasaltAPIError,
    NotFoundError,
    UnauthorizedError,
)
from basalt.prompts.client import PromptsClient
from basalt.prompts.models import Prompt
from basalt.utils.memcache import MemoryCache

# Add parent directory to path to import basalt
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

os.environ["BASALT_BUILD"] = "development"


def initialize_client() -> PromptsClient:
    """Initialize the Basalt Prompts client with API key and caching.

    Returns:
        PromptsClient: Configured prompts client instance.

    Raises:
        ValueError: If BASALT_API_KEY environment variable is not set.
    """
    api_key = os.getenv("BASALT_API_KEY")
    if not api_key:
        raise ValueError("BASALT_API_KEY environment variable not set")

    # Setup caching and logging
    cache = MemoryCache()
    fallback_cache = MemoryCache()  # Infinite cache for fallback

    logging.basicConfig(level=logging.INFO)

    return PromptsClient(
        api_key=api_key,
        cache=cache,
        fallback_cache=fallback_cache,
    )


def example_1_list_prompts(client: PromptsClient) -> None:
    """Example 1: List all prompts synchronously."""
    logging.info("Listing prompts synchronously")

    try:
        prompt_list = client.list_sync()

        if prompt_list:
            for _i, _prompt in enumerate(prompt_list[:3], 1):
                logging.info(f"{_i}. {_prompt.slug} - {_prompt.name}" + "\n")
            if len(prompt_list) > 3:
                pass
    except UnauthorizedError:
        logging.error("Unauthorized access")
    except BasaltAPIError:
        logging.error("Basalt API error occurred")


def example_2_get_prompt(client: PromptsClient) -> None:
    """Example 2: Get a specific prompt synchronously."""
    logging.info("Getting a specific prompt synchronously")

    prompt_slug = os.getenv("BASALT_TEST_PROMPT_SLUG")
    if not prompt_slug:
        logging.error("BASALT_TEST_PROMPT_SLUG environment variable not set")
        return

    try:
        prompt = client.get_sync(prompt_slug)
        logging.info(prompt.text[:100] + "..." if len(prompt.text) > 100 else prompt.text + "\n")
    except NotFoundError:
        logging.error("Prompt not found")
    except UnauthorizedError:
        logging.error("Unauthorized access")
    except BasaltAPIError:
        logging.error("Basalt API error occurred")


def example_3_get_prompt_with_version(client: PromptsClient) -> None:
    """Example 3: Get a prompt with a specific version."""
    logging.info("Getting a prompt with a specific version")

    prompt_slug = os.getenv("BASALT_TEST_PROMPT_SLUG")
    if not prompt_slug:
        logging.error("BASALT_TEST_PROMPT_SLUG environment variable not set")
        return

    try:
        # First describe to get available versions
        describe = client.describe_sync(prompt_slug)

        if describe.available_versions:
            version = describe.available_versions[0]
            prompt = client.get_sync(prompt_slug, version=version)
            logging.info(prompt.text[:80] + "..." if len(prompt.text) > 80 else prompt.text)
            logging.info(f"Using version: {version} \n\n")
        else:
            logging.error("No available versions found")
    except NotFoundError:
        logging.error("Prompt not found", exc_info=True)
    except UnauthorizedError:
        logging.error("Unauthorized access", exc_info=True)
    except BasaltAPIError:
        logging.error("Basalt API error occurred", exc_info=True)



def example_4_get_prompt_with_tag(client: PromptsClient) -> None:
    """Example 4: Get a prompt with a specific tag."""
    logging.info("Getting a prompt with a specific tag")

    prompt_slug = os.getenv("BASALT_TEST_PROMPT_SLUG")
    if not prompt_slug:
        logging.error("BASALT_TEST_PROMPT_SLUG environment variable not set")
        return

    try:
        # First describe to get available tags
        describe = client.describe_sync(prompt_slug)
        logging.info(f"Available tags: {describe.available_tags}")

        if describe.available_tags:
            tag = describe.available_tags[0]
            client.get_sync(prompt_slug, tag=tag)
            logging.info(f"Using tag: {tag} \n\n")
        else:
            logging.warning("No available tags found")

    except NotFoundError:
        logging.error("Prompt not found", exc_info=True)
    except BasaltAPIError:
        logging.error("Basalt API error occurred", exc_info=True)

def example_5_get_prompt_with_variables(client: PromptsClient) -> None:
    """Example 5: Get a prompt with variable substitution."""
    logging.info("Getting a prompt with variable substitution")

    prompt_slug = os.getenv("BASALT_TEST_PROMPT_SLUG")
    if not prompt_slug:
        return

    try:
        # First describe to check for variables
        describe = client.describe_sync(prompt_slug)
        logging.info(f"Available variables: {describe.variables}")

        if describe.variables:
            # Create variables dict from available variables
            variables = {}
            for var in describe.variables:
                # Variables might be dicts with 'name' key, or strings
                for element in var:
                    variables[element] = f"test_value_{element}"

            if variables:
                prompt = client.get_sync(prompt_slug, variables=variables)
                logging.info(prompt.text[:80] + "..." if len(prompt.text) > 80 else prompt.text + "\n\n")
        else:
            logging.warning("No variables found")
    except NotFoundError:
        logging.error("Prompt not found", exc_info=True)
    except BasaltAPIError:
        logging.error("Basalt API error occurred", exc_info=True)


def example_6_describe_prompt(client: PromptsClient) -> None:
    """Example 6: Describe a prompt to get metadata."""
    logging.info("Describing a prompt to get metadata")

    prompt_slug = os.getenv("BASALT_TEST_PROMPT_SLUG")
    if not prompt_slug:
        return

    try:
        response = client.describe_sync(prompt_slug)
        logging.info(f"Prompt description: {response}")

        if response.variables:
            # Extract variable names safely
            var_names = []
            for v in response.variables:
                if isinstance(v, dict):
                    var_names.append(v)
            if var_names:
                logging.info(f"Variables: {var_names}\n\n")
    except NotFoundError:
        logging.error("Prompt not found", exc_info=True)
    except BasaltAPIError:
        logging.error("Basalt API error occurred", exc_info=True)


async def example_7_async_list_prompts(client: PromptsClient) -> None:
    """Example 7: List all prompts asynchronously."""

    try:
        prompt_list = await client.list()

        if prompt_list:
            for _i, _prompt in enumerate(prompt_list[:3], 1):
                pass
            if len(prompt_list) > 3:
                pass
    except UnauthorizedError:
        logging.error("Unauthorized access")
    except BasaltAPIError:
        logging.error("Basalt API error occurred", exc_info=True)


async def example_8_async_get_prompt(client: PromptsClient) -> None:
    """Example 8: Get a prompt asynchronously."""

    prompt_slug = os.getenv("BASALT_TEST_PROMPT_SLUG")
    if not prompt_slug:
        return

    try:
        await client.get(prompt_slug)
    except NotFoundError:
        logging.error("Prompt not found", exc_info=True)
    except BasaltAPIError:
        logging.error("Basalt API error occurred", exc_info=True)


async def example_9_concurrent_operations(client: PromptsClient) -> None:
    """Example 9: Execute multiple async operations concurrently.

    Demonstrates concurrent execution of multiple API calls with proper
    error handling and type-safe result processing.
    """
    logging.info("Executing concurrent operations")

    prompt_slug = os.getenv("BASALT_TEST_PROMPT_SLUG")
    if not prompt_slug:
        logging.error("BASALT_TEST_PROMPT_SLUG environment variable not set")
        return

    try:
        # Create multiple concurrent tasks
        tasks = [
            client.list(),
            client.get(prompt_slug),
            client.describe(prompt_slug),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Type-safe extraction of results
        list_result = results[0]
        get_result = results[1]
        describe_result = results[2]

        # Process list result
        if isinstance(list_result, Exception):
            logging.error(f"List operation failed: {type(list_result).__name__}: {list_result}")
        elif isinstance(list_result, list):
            logging.info(f"Retrieved {len(list_result)} prompts")
        else:
            logging.warning(f"Unexpected list result type: {type(list_result)}")

        # Process get result
        if isinstance(get_result, Exception):
            logging.error(f"Get operation failed: {type(get_result).__name__}: {get_result}")
        elif isinstance(get_result, Prompt):
            logging.info(f"Retrieved prompt: {get_result.slug} (version: {get_result.version})")
        else:
            logging.warning(f"Unexpected get result type: {type(get_result)}")

        # Process describe result
        if isinstance(describe_result, Exception):
            logging.error(f"Describe operation failed: {type(describe_result).__name__}: {describe_result}")
        else:
            logging.info(f"Retrieved prompt description for: {describe_result}")

    except BasaltAPIError as e:
        logging.error(f"Basalt API error during concurrent operations: {e}", exc_info=True)
    except Exception as e:
        logging.error(f"Unexpected error during concurrent operations: {type(e).__name__}: {e}", exc_info=True)


async def run_async_examples(client: PromptsClient) -> None:
    """Run all async examples."""
    await example_7_async_list_prompts(client)
    await example_8_async_get_prompt(client)
    await example_9_concurrent_operations(client)


def main() -> None:
    """Run all examples."""

    try:
        # Initialize client
        client = initialize_client()

        # Run synchronous examples
        example_1_list_prompts(client)
        example_2_get_prompt(client)
        example_3_get_prompt_with_version(client)
        example_4_get_prompt_with_tag(client)
        example_5_get_prompt_with_variables(client)
        example_6_describe_prompt(client)

        # Run asynchronous examples
        asyncio.run(run_async_examples(client))

    except ValueError:
        logging.error("API key not set in environment variables")
        sys.exit(1)
    except Exception:
        logging.error("An unexpected error occurred", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
