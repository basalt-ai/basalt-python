import asyncio
import os

from mistralai.client import MistralClient

from basalt import Basalt
from basalt.observability import ObserveKind, observe

# Ensure API keys are set
if "BASALT_API_KEY" not in os.environ:
    os.environ["BASALT_API_KEY"] = "test-key"
if "MISTRAL_API_KEY" not in os.environ:
    pass

# Initialize Basalt
client = Basalt(
    api_key=os.environ["BASALT_API_KEY"],
    observability_metadata={
        "env": "staging",
        "provider": "mistral",
        "example": "manual-instrumentation"
    }
)

# Initialize Mistral client
# Note: We assume auto-instrumentation is NOT supported or disabled for Mistral in this example.
mistral_client = MistralClient(api_key=os.environ.get("MISTRAL_API_KEY", "dummy"))

async def run_mistral_flow(topic: str):
    # 1. Workflow Span
    with observe(name="mistral_workflow", kind="workflow"):
        observe.identify(user="user_123")
        observe.input({"topic": topic})


        # 2. Manual LLM Span
        # Since we are not using auto-instrumentation, we manually wrap the call
        # and provide the necessary metadata (model, input, output).
        with observe(name="mistral_chat", kind=ObserveKind.GENERATION) as llm_span:
            # Allow the user to pick a model via env var, or fall back to a safe default.
            # Common model names from the Mistral SDK are: "mistral-small-latest", "mistral-tiny", "open-mistral-7b".
            # Set MISTRAL_MODEL to one of these values (or a provider-specific model id) to change selection.
            raw_model = os.environ.get("MISTRAL_MODEL", "")
            model_aliases = {
                "small": "mistral-small-latest",
                "tiny": "mistral-tiny",
                "open-7b": "open-mistral-7b",
                "7b": "open-mistral-7b",
            }

            if raw_model:
                model = model_aliases.get(raw_model, raw_model)
            else:
                model = "mistral-small-latest"

            # Use a plain dict list for messages (the Python SDK accepts this format)
            messages = [{"role": "user", "content": f"Explain {topic} in one sentence."}]

            # Set LLM Request Metadata manually
            llm_span.set_input({"messages": messages})
            llm_span.set_attribute("llm.model", model)
            llm_span.set_attribute("llm.provider", "mistral")


            # Real Call (use the synchronous completion method from the SDK)
            # If you prefer async, call `chat.complete_async` from within an async client context.
            chat_response = mistral_client.chat.complete(
                model=model,
                messages=messages,
                stream=False,
            )

            content = chat_response.choices[0].message.content

            # Set LLM Response Metadata manually
            llm_span.set_output(content)
            llm_span.set_attribute("llm.usage.total_tokens", chat_response.usage.total_tokens)

            return content

if __name__ == "__main__":
    try:
        # Using asyncio.run for the async flow, though MistralClient can be sync.
        # This example assumes an async flow structure for demonstration.
        result = asyncio.run(run_mistral_flow("Quantum Physics"))
    except Exception:
        pass
