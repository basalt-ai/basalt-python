import json
import os

from openai import OpenAI

from basalt import Basalt
from basalt.observability import ObserveKind, evaluate, observe

# Ensure API keys are set
if "BASALT_API_KEY" not in os.environ:
    os.environ["BASALT_API_KEY"] = "test-key"
if "OPENAI_API_KEY" not in os.environ:
    pass
    # We don't exit to allow syntax checking, but real run needs key

# Initialize Basalt
# Auto-instrumentation for OpenAI is enabled by default when the library is installed.
client = Basalt(
    api_key=os.environ["BASALT_API_KEY"],
    observability_metadata={
        "env": "development",
        "provider": "openai",
        "example": "auto-instrumentation"
    }
)

# Initialize OpenAI client
openai_client = OpenAI()

# Mock Tool (RAG/Search)
@observe(kind=ObserveKind.TOOL, name="get_weather")
def get_weather(location: str):
    """Mock weather tool."""
    return json.dumps({"location": location, "temperature": "22C", "condition": "Sunny"})

@evaluate("helpfulness")
@observe(name="weather_assistant")
def run_weather_assistant(user_query: str):
    observe.identify(user="user_123")

    # 1. Mock Tool Call (simulating a decision to call a tool)
    weather_data = get_weather("San Francisco, CA")

    # 2. Real LLM Call (Auto-instrumented)
    # Basalt automatically captures the span, input (messages), and output (content).
    response = openai_client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "You are a helpful weather assistant."},
            {"role": "user", "content": f"Context: {weather_data}\n\nQuery: {user_query}"}
        ]
    )

    content = response.choices[0].message.content

    return content

if __name__ == "__main__":
    try:
        result = run_weather_assistant("What's the weather like in SF?")
    except Exception:
        pass
