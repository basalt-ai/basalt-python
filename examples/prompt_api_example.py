"""Basalt Prompt SDK Example with OpenTelemetry.

This script demonstrates how to use the Basalt Prompt SDK with OpenTelemetry tracing.
It showcases:
- Automatic tracing of API calls
- Manual tracing with custom spans
- Both synchronous and asynchronous operations
- Proper error handling and telemetry shutdown

Before running this script:
1. Set your API key: export BASALT_API_KEY="your-api-key"
2. Set a test prompt slug: export BASALT_TEST_PROMPT_SLUG="your-prompt-slug"
3. Install dependencies: pip install basalt-sdk

OpenTelemetry Configuration:
- Traces are sent to Basalt's OTEL collector by default
- Production endpoint: https://otel.getbasalt.ai/v1/traces
- Development endpoint: http://localhost:4318/v1/traces
- Configure custom endpoints via BASALT_OTEL_EXPORTER_OTLP_ENDPOINT env var

Reference: See MIGRATION_GUIDE.md for details on the new exception-based API.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

from basalt import Basalt, TelemetryConfig
from basalt.observability.context_managers import trace_span
from basalt.prompts.models import Prompt
from basalt.types.exceptions import (
    BasaltAPIError,
    NotFoundError,
    UnauthorizedError,
)

# Add parent directory to path to import basalt
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

os.environ["BASALT_BUILD"] = "development"


def initialize_client() -> Basalt:
    """Initialize the Basalt client with API key and OpenTelemetry.

    Returns:
        Basalt: Configured Basalt client instance with telemetry enabled.

    Raises:
        ValueError: If BASALT_API_KEY environment variable is not set.
    """
    api_key = os.getenv("BASALT_API_KEY")
    if not api_key:
        raise ValueError("BASALT_API_KEY environment variable not set")

    logging.basicConfig(level=logging.INFO)

    # Configure telemetry with OpenTelemetry
    telemetry_config = TelemetryConfig(
        service_name="prompt-api-example",
        environment="development",
        enable_llm_instrumentation=True,
        llm_trace_content=True,  # Include prompt/completion content in traces
    )

    # Initialize Basalt client with telemetry
    return Basalt(
        api_key=api_key,
        telemetry_config=telemetry_config,
    )


def example_1_list_prompts(client: Basalt) -> None:
    """Example 1: List all prompts synchronously."""
    logging.info("Listing prompts synchronously")

    try:
        prompt_list = client.prompts.list_sync()

        if prompt_list:
            for _i, _prompt in enumerate(prompt_list[:3], 1):
                logging.info(f"{_i}. {_prompt.slug} - {_prompt.name}" + "\n")
            if len(prompt_list) > 3:
                pass
    except UnauthorizedError:
        logging.error("Unauthorized access")
    except BasaltAPIError:
        logging.error("Basalt API error occurred")


def example_2_get_prompt(client: Basalt) -> None:
    """Example 2: Get a specific prompt synchronously."""
    logging.info("Getting a specific prompt synchronously")

    prompt_slug = os.getenv("BASALT_TEST_PROMPT_SLUG")
    if not prompt_slug:
        logging.error("BASALT_TEST_PROMPT_SLUG environment variable not set")
        return

    try:
        prompt = client.prompts.get_sync(prompt_slug)
        logging.info(prompt.text[:100] + "..." if len(prompt.text) > 100 else prompt.text + "\n")
    except NotFoundError:
        logging.error("Prompt not found")
    except UnauthorizedError:
        logging.error("Unauthorized access")
    except BasaltAPIError:
        logging.error("Basalt API error occurred")


def example_3_get_prompt_with_version(client: Basalt) -> None:
    """Example 3: Get a prompt with a specific version."""
    logging.info("Getting a prompt with a specific version")

    prompt_slug = os.getenv("BASALT_TEST_PROMPT_SLUG")
    if not prompt_slug:
        logging.error("BASALT_TEST_PROMPT_SLUG environment variable not set")
        return

    try:
        # First describe to get available versions
        describe = client.prompts.describe_sync(prompt_slug)

        if describe.available_versions:
            version = describe.available_versions[0]
            prompt = client.prompts.get_sync(prompt_slug, version=version)
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



def example_4_get_prompt_with_tag(client: Basalt) -> None:
    """Example 4: Get a prompt with a specific tag."""
    logging.info("Getting a prompt with a specific tag")

    prompt_slug = os.getenv("BASALT_TEST_PROMPT_SLUG")
    if not prompt_slug:
        logging.error("BASALT_TEST_PROMPT_SLUG environment variable not set")
        return

    try:
        # First describe to get available tags
        describe = client.prompts.describe_sync(prompt_slug)
        logging.info(f"Available tags: {describe.available_tags}")

        if describe.available_tags:
            tag = describe.available_tags[0]
            client.prompts.get_sync(prompt_slug, tag=tag)
            logging.info(f"Using tag: {tag} \n\n")
        else:
            logging.warning("No available tags found")

    except NotFoundError:
        logging.error("Prompt not found", exc_info=True)
    except BasaltAPIError:
        logging.error("Basalt API error occurred", exc_info=True)

def example_5_get_prompt_with_variables(client: Basalt) -> None:
    """Example 5: Get a prompt with variable substitution."""
    logging.info("Getting a prompt with variable substitution")

    prompt_slug = os.getenv("BASALT_TEST_PROMPT_SLUG")
    if not prompt_slug:
        return

    try:
        # First describe to check for variables
        describe = client.prompts.describe_sync(prompt_slug)
        logging.info(f"Available variables: {describe.variables}")

        if describe.variables:
            # Create variables dict from available variables
            variables = {}
            for var in describe.variables:
                # Variables might be dicts with 'name' key, or strings
                for element in var:
                    variables[element] = f"test_value_{element}"

            if variables:
                prompt = client.prompts.get_sync(prompt_slug, variables=variables)
                logging.info(prompt.text[:80] + "..." if len(prompt.text) > 80 else prompt.text + "\n\n")
        else:
            logging.warning("No variables found")
    except NotFoundError:
        logging.error("Prompt not found", exc_info=True)
    except BasaltAPIError:
        logging.error("Basalt API error occurred", exc_info=True)


def example_6_describe_prompt(client: Basalt) -> None:
    """Example 6: Describe a prompt to get metadata."""
    logging.info("Describing a prompt to get metadata")

    prompt_slug = os.getenv("BASALT_TEST_PROMPT_SLUG")
    if not prompt_slug:
        return

    try:
        response = client.prompts.describe_sync(prompt_slug)
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


def example_6b_publish_prompt(client: Basalt) -> None:
    """Example 6b: Publish a prompt with a new deployment tag."""
    logging.info("Publishing a prompt with a new deployment tag")

    prompt_slug = os.getenv("BASALT_TEST_PROMPT_SLUG")
    if not prompt_slug:
        logging.error("BASALT_TEST_PROMPT_SLUG environment variable not set")
        return

    try:
        # First describe to get available versions
        describe = client.prompts.describe_sync(prompt_slug)

        if describe.available_versions:
            version = describe.available_versions[0]

            # Generate a unique tag name
            import time
            new_tag = f"example-deployment-{int(time.time())}"

            # Publish the prompt
            response = client.prompts.publish_prompt_sync(
                slug=prompt_slug,
                new_tag=new_tag,
                version=version,
            )

            logging.info(f"Published prompt with tag: {response.label}")
            logging.info(f"Deployment tag ID: {response.id}\n\n")
        else:
            logging.warning("No available versions found for publishing")

    except NotFoundError:
        logging.error("Prompt not found", exc_info=True)
    except BasaltAPIError:
        logging.error("Basalt API error occurred", exc_info=True)


def example_7_manual_tracing(client: Basalt) -> None:
    """Example 7: Manual tracing with custom spans.

    Demonstrates how to use OpenTelemetry context managers for custom tracing.
    """
    logging.info("Demonstrating manual tracing with custom spans")

    prompt_slug = os.getenv("BASALT_TEST_PROMPT_SLUG")
    if not prompt_slug:
        logging.warning("BASALT_TEST_PROMPT_SLUG not set, using a default value for demo")
        prompt_slug = "example-prompt"

    # Create a custom span to trace a business operation
    with trace_span("process_prompt_workflow", attributes={"workflow.type": "example"}) as span:
        span.add_event("workflow_started")
        span.set_attribute("prompt.slug", prompt_slug)

        try:
            # This operation is automatically traced by the SDK
            prompt = client.prompts.get_sync(prompt_slug)

            span.add_event("prompt_retrieved", attributes={
                "prompt.version": prompt.version,
                "prompt.length": len(prompt.text)
            })

            # Simulate some processing
            logging.info(f"Processing prompt: {prompt.slug}")
            span.set_attribute("processing.status", "completed")

        except NotFoundError:
            span.add_event("prompt_not_found")
            span.set_attribute("processing.status", "failed")
            logging.warning(f"Prompt '{prompt_slug}' not found")
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("processing.status", "error")
            logging.error(f"Error processing prompt: {e}")


async def example_8_async_list_prompts(client: Basalt) -> None:
    """Example 8: List all prompts asynchronously."""

    try:
        prompt_list = await client.prompts.list()

        if prompt_list:
            for _i, _prompt in enumerate(prompt_list[:3], 1):
                pass
            if len(prompt_list) > 3:
                pass
    except UnauthorizedError:
        logging.error("Unauthorized access")
    except BasaltAPIError:
        logging.error("Basalt API error occurred", exc_info=True)


async def example_9_async_get_prompt(client: Basalt) -> None:
    """Example 9: Get a prompt asynchronously."""

    prompt_slug = os.getenv("BASALT_TEST_PROMPT_SLUG")
    if not prompt_slug:
        return

    try:
        await client.prompts.get(prompt_slug)
    except NotFoundError:
        logging.error("Prompt not found", exc_info=True)
    except BasaltAPIError:
        logging.error("Basalt API error occurred", exc_info=True)


async def example_10_concurrent_operations(client: Basalt) -> None:
    """Example 10: Execute multiple async operations concurrently.

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
            client.prompts.list(),
            client.prompts.get(prompt_slug),
            client.prompts.describe(prompt_slug),
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


async def run_async_examples(client: Basalt) -> None:
    """Run all async examples."""
    await example_8_async_list_prompts(client)
    await example_9_async_get_prompt(client)
    await example_10_concurrent_operations(client)


def main() -> None:
    """Run all examples."""

    client = None
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
        example_6b_publish_prompt(client)
        example_7_manual_tracing(client)

        # Run asynchronous examples
        asyncio.run(run_async_examples(client))

    except ValueError:
        logging.error("API key not set in environment variables")
        sys.exit(1)
    except Exception:
        logging.error("An unexpected error occurred", exc_info=True)
        sys.exit(1)
    finally:
        # Ensure telemetry data is flushed before exit
        if client:
            logging.info("Shutting down client and flushing telemetry data...")
            client.shutdown()


if __name__ == "__main__":
    main()
