"""
Example demonstrating the context manager pattern for prompts.

This example shows how to use prompts.get_sync() as a context manager
to automatically scope auto-instrumented GenAI calls.
"""
import os

from basalt import Basalt

# Initialize Basalt client
client = Basalt(
    api_key=os.environ.get("BASALT_API_KEY", "test-key"),
)

# Example 1: Context Manager Pattern (Recommended for Observability)
# This creates a span that scopes any LLM calls within the context

with client.prompts.get_sync("qa-prompt", variables={"context": "Paris is the capital of France"}) as prompt:

    # Any auto-instrumented LLM calls here would automatically nest
    # under the prompt span in the trace
    # Example:
    # response = openai.chat.completions.create(
    #     model=prompt.model.model,
    #     messages=[{"role": "user", "content": prompt.text}]
    # )

    pass


# Example 2: Imperative Pattern (Backward Compatible, Always Instrumented)
# This creates a span that immediately ends

prompt = client.prompts.get_sync("qa-prompt", variables={"context": "London is the capital of UK"})

# LLM calls here create their own separate spans
# Example:
# response = openai.chat.completions.create(
#     model=prompt.model.model,
#     messages=[{"role": "user", "content": prompt.text}]
# )



# Example 3: Async Context Manager Pattern


async def async_example():
    async with await client.prompts.get("qa-prompt", variables={"context": "Berlin is the capital of Germany"}):

        # Async LLM calls would nest under this span
        # Example:
        # async with httpx.AsyncClient() as http_client:
        #     response = await openai_async.chat.completions.create(...)

        pass

# Uncomment to run async example:
# asyncio.run(async_example())

