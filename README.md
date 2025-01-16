# Basalt SDK

Basalt is a powerful tool for managing AI prompts and their release workflows. This SDK is the official Python package for interacting with your Basalt prompts.

## Installation

Install the Basalt SDK via pip:

```bash
pip install basalt-sdk
```

## Usage

### Importing and Initializing the SDK

To get started, import the `PromptSDK` class and initialize it with your API and cache instances:

```python
from basalt import Basalt

basalt = Basalt(api_key="my-dev-api-key")

# Specify a log_level
basalt = Basalt(api_key="my-dev-api-key", log_level="none")

# Or with an env
import os

basalt = Basalt(api_key=os.getenv("BASALT_API_KEY"))
```

### Available Methods

#### Prompts
Your Basalt instance exposes a `prompt` property for interacting with your Basalt prompts:

- **Get a Prompt**

  Retrieve a specific prompt using a slug, and optional filters `tag` and `version`. Without tag or version, the production version of your prompt is selected by default.

  **Example Usage:**

  ```python
  error, result = basalt.prompt.get('prompt-slug')

  # With optional tag or version parameters
  error, result = basalt.get(slug='prompt-slug', tag='latest')
  error, result = basalt.get(slug='prompt-slug', version='1.0.0')

  # If your prompt has variables,
  # pass them when fetching your prompt
  error, result = basale.get(slug='prompt-slug', variables={ name: 'John Doe' })

  # Handle the result by unwrapping the error / value
  if error:
      print('Could not fetch prompt', error)
  else:
      # Use the prompt with your AI provider of choice
      # Example: OpenAI
      openai_client.chat_completion.create(
          model='gpt-4',
          messages=[{'role': 'user', 'content': result.prompt}]
      )
  ```

## License

[TODO]