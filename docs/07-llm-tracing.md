# LLM & Generation Tracing

Specialized tracing for LLM API calls with automatic capture of prompts, completions, and token usage.

> **Important:** If your LLM is automatically instrumented (for example, by importing the OpenAI instrumentation in your telemetry config), you should **not** manually instrument LLM calls. Manual instrumentation is only needed if:
> - You cannot or do not want to use automatic instrumentation, or
> - You are using an LLM provider not yet supported by automatic instrumentation.

## Overview

Basalt provides specialized spans for LLM operations that automatically capture:
- Model provider and name
- Prompts and completions (when enabled)
- Token usage (input, output, total)
- Model parameters (temperature, max_tokens, etc.)

## Using the Decorator

```python
from basalt.observability import observe_generation

@observe_generation(name="llm.generate_summary")
def generate_summary(text: str):
    response = llm_client.generate(text)
    return response
```

## Using Context Manager

```python
from basalt.observability import trace_generation

with trace_generation(name="llm.chat") as span:
    span.set_model("gpt-4")
    span.set_prompt("Tell me a joke")

    response = openai_client.chat.completions.create(...)

    span.set_completion(response.choices[0].message.content)
    span.set_tokens(
        input=response.usage.prompt_tokens,
        output=response.usage.completion_tokens
    )
```

## LLMSpanHandle Methods

Additional methods for LLM spans:

- `set_model(model)` - Set model name
- `set_prompt(prompt)` - Set input prompt
- `set_completion(completion)` - Set model output
- `set_tokens(input, output, total)` - Set token counts
- `add_evaluator(slug)` - Attach evaluator

