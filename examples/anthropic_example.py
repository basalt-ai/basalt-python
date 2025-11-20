import os

from anthropic import Anthropic

from basalt import Basalt
from basalt.observability import observe

# Ensure API keys are set
if "BASALT_API_KEY" not in os.environ:
    os.environ["BASALT_API_KEY"] = "test-key"
if "ANTHROPIC_API_KEY" not in os.environ:
    pass

# Initialize Basalt
# Auto-instrumentation for Anthropic is enabled by default.
client = Basalt(
    api_key=os.environ["BASALT_API_KEY"],
    observability_metadata={
        "env": "production",
        "provider": "anthropic",
        "example": "auto-instrumentation"
    }
)

# Initialize Anthropic client
anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

@observe(name="claude_analysis")
def analyze_text(text: str):
    observe.identify(organization="org_999", user="analyst_01")
    observe.input({"text": text})

    # Real LLM Call (Auto-instrumented)
    message = anthropic_client.messages.create(
        model="claude-4-haiku",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": f"Analyze the sentiment of this text: {text}"}
        ]
    )

    content = message.content[0].text

    return content

if __name__ == "__main__":
    try:
        result = analyze_text("I love using Basalt for observability! It makes debugging so easy.")
    except Exception:
        pass
