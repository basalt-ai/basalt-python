import os
import sys
import time
from typing import Dict, Any

# Add the parent directory to the path so we can import the basalt package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basalt.sdk.monitorsdk import MonitorSDK
from basalt.sdk.promptsdk import PromptSDK
from basalt.utils.api import Api
from basalt.utils.logger import Logger
from basalt.utils.networker import Networker

class MockOpenAI:
    """Mock OpenAI client for demonstration purposes."""
    
    def generate_text(self, prompt: str) -> str:
        """Generate text using a mock OpenAI."""
        print(f"Generating text with mock OpenAI for prompt: {prompt}")
        return f"Generated response for: {prompt[:50]}..."
    
    def classify_content(self, content: str) -> str:
        """Classify content using a mock OpenAI."""
        print(f"Classifying content: {content[:50]}...")
        return "Classification: Technology, Healthcare, AI"
    
    def translate_text(self, text: str) -> str:
        """Translate text using a mock OpenAI."""
        print(f"Translating: {text[:50]}...")
        return "Traducción: Este es un texto traducido al español."
    
    def summarize_text(self, text: str) -> str:
        """Summarize text using a mock OpenAI."""
        print(f"Summarizing: {text[:50]}...")
        return "Summary: This is a concise summary of the provided content."

class MockPromptSDK:
    """Mock Prompt SDK for demonstration purposes."""
    
    def get(self, slug: str, **kwargs) -> Dict[str, Any]:
        """Get a prompt from the mock Prompt SDK."""
        print(f"Getting prompt {slug} with variables: {kwargs.get('variables', {})}")
        
        # Return a mock prompt response
        return {
            "value": {
                "text": f"This is a prompt for {slug} with variables: {kwargs.get('variables', {})}"
            },
            "generation": {
                "prompt": {
                    "slug": slug,
                    "model": "gpt-3.5-turbo"
                }
            }
        }

class Basalt:
    """Mock Basalt SDK for demonstration purposes."""
    
    def __init__(self, api_key: str):
        networker = Networker()
        logger = Logger()
        self.api = Api(
            networker=networker,
            root_url="http://localhost:3001",
            api_key=api_key,
            sdk_version="0.1.0",
            sdk_type="py-test"
        )
        self.monitor = MonitorSDK(self.api, logger)
        self.prompt = MockPromptSDK()

# def test_simple_monitor_sdk():
#     """Test the monitor SDK implementation."""
#     print("Starting Basalt monitoring test...")
    
#     # Initialize clients
#     basalt = Basalt(api_key="sk-ba4df805e4cc25cbfedf2fc53d1168526888c00c9026ba3cfcb178bbb444eb16")
#     openai = MockOpenAI()
    
#     # Create a user and content
#     user = {"id": "user123", "name": "John Doe"}
#     content = "Create a technical article about machine learning applications in healthcare"
    
#     # Create a main trace for the entire user request
#     print(f'Creating trace for: {content[:50]}...')

    
#     try:
#         # Get prompt from Basalt
#         basalt_prompt = basalt.prompt.get("generate-test-cases",
#             variables={"promptName": "translation", "promptVariables": "spanish"},
#             version="0.8"
#         )
        
#     except Exception as e:
#         # Log any uncaught errors to the main trace
#         main_trace.update({
#             "metadata": {
#                 **(main_trace.metadata or {}),
#                 "error": {
#                     "name": "ProcessingError",
#                     "message": str(e),
#                     "stack": getattr(e, "__traceback__", None)
#                 }
#             }
#         })
        
#         print(f"Error processing user content: {e}")
#     finally:
#         # Complete the main trace
#         main_trace.end()
    
#     print("Test completed successfully!")
#     return main_trace

def test_monitor_sdk():
    """Test the monitor SDK implementation."""
    print("Starting Basalt monitoring test...")
    
    # Initialize clients
    basalt = Basalt(api_key="sk-ba4df805e4cc25cbfedf2fc53d1168526888c00c9026ba3cfcb178bbb444eb16")
    openai = MockOpenAI()
    
    # Create a user and content
    user = {"id": "user123", "name": "John Doe"}
    content = "Create a technical article about machine learning applications in healthcare"
    
    # Create a main trace for the entire user request
    print(f'Creating trace for: {content[:50]}...')
    main_trace = basalt.monitor.create_trace(
        "theo-slug",
        {
            "input": content,
            "user": user,
            "organization": {"id": "org-123", "name": "Basalt"},
            "metadata": {"property1": "value1", "property2": "value2"},
            "name": "User Content Processing"
        }
    )
    
    try:
        # Step 1: Content generation
        generation_span = main_trace.create_log({
            "type": "span",
            "name": "content-generation",
            "input": content,
            "metadata": {"property1": "value1", "property2": "value2"}
        })
        
        # Get prompt from Basalt
        basalt_prompt = basalt.prompt.get("generate-test-cases", 
            variables={"promptName": "content-generation", "promptVariables": "user-content"},
            version="0.8"
        )
        
        # Create generation log
        generation_log = generation_span.create_generation({
            "name": "text-generation",
            "input": content,
            "metadata": {"property1": "value1", "property2": "value2"},
            "prompt": basalt_prompt["generation"]["prompt"],
            "variables": {"article_subject": content}
        })
        
        # Update metadata
        generation_log.update({
            "metadata": {
                **(generation_log.metadata or {}),
                "contentLength": len(content),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        })
        
        # Generate text using OpenAI
        generated_text = openai.generate_text(basalt_prompt["value"]["text"])
        
        # Update generation with output and metrics
        generation_log.update({
            "metadata": {
                **(generation_log.metadata or {}),
                "contentLength": len(generated_text),
                "processingTime": 500  # Simulated processing time
            }
        })
        
        generation_log.update({
            "output": generated_text,
            "prompt": basalt_prompt["generation"]["prompt"]
        })
        
        generation_span.end(generated_text)
        
        # Step 2: Classification
        classification_span = main_trace.create_log({
            "type": "span",
            "name": "classification",
            "input": generated_text
        })
        
        # Get prompt from Basalt
        basalt_prompt = basalt.prompt.get("generate-test-cases",
            variables={"promptName": "classification", "promptVariables": "content-categories"},
            version="0.8"
        )
        
        # Create generation log
        class_gen = classification_span.create_generation({
            "name": "content-classification",
            "input": generated_text,
            "prompt": basalt_prompt["generation"]["prompt"],
            "variables": {"article_content": generated_text}
        })
        
        # Classify content
        categories = openai.classify_content(generated_text)
        
        # Update generation with output
        class_gen.update({
            "output": categories,
            "prompt": basalt_prompt["generation"]["prompt"]
        })
        
        classification_span.end(categories)
        
        # Step 3: Translation
        translation_span = main_trace.create_log({
            "type": "span",
            "name": "translation",
            "input": generated_text
        })
        
        # Get prompt from Basalt
        basalt_prompt = basalt.prompt.get("generate-test-cases",
            variables={"promptName": "translation", "promptVariables": "spanish"},
            version="0.8"
        )
        
        # Create generation log
        trans_gen = translation_span.create_generation({
            "name": "content-translation",
            "input": generated_text,
            "prompt": basalt_prompt["generation"]["prompt"],
            "variables": {"text_to_translate": generated_text, "language": "Spanish"}
        })
        
        # Create nested logs
        inner_span = translation_span.create_log({
            "type": "span",
            "name": "content-translation-2",
            "input": generated_text
        })
        
        inner_span.create_log({
            "type": "span",
            "name": "content-translation-3",
            "input": generated_text
        })
        
        # Translate text
        translated_text = openai.translate_text(basalt_prompt["value"]["text"])
        
        # Update generation with output
        trans_gen.update({
            "input": generated_text,
            "output": translated_text[:100]  # First 100 chars
        })
        
        translation_span.end(translated_text)
        
        # Step 4: Summarization
        summary_span = main_trace.create_log({
            "type": "span",
            "name": "summarization",
            "input": generated_text
        })
        
        # Get prompt from Basalt
        basalt_prompt = basalt.prompt.get("generate-test-cases",
            variables={"promptName": "summarization", "promptVariables": "content-summary"},
            version="0.8"
        )
        
        # Create generation log
        summary_gen = summary_span.create_generation({
            "name": "content-summary",
            "input": generated_text,
            "prompt": basalt_prompt["generation"]["prompt"],
            "variables": {"article_content": generated_text}
        })
        
        # Summarize text
        summary = openai.summarize_text(generated_text)
        
        # Update generation with output and metrics
        summary_gen.update({
            "metadata": {
                **(summary_gen.metadata or {}),
                "compressionRatio": len(summary) / len(generated_text)
            }
        })
        
        summary_gen.update({"output": summary})
        main_trace.update({"output": summary})
        summary_span.end(summary)
        
        # Add overall metrics to the main trace
        main_trace.update({
            "metadata": {
                **(main_trace.metadata or {}),
                "userId": user["id"],
                "processingSteps": 4,
                "totalContentLength": len(generated_text),
                "status": "completed"
            }
        })
        
        # Log the final result
        print(f"Processing complete for user {user['id']}")
        print(f"Summary: {summary[:100]}...")
        
    except Exception as e:
        # Log any uncaught errors to the main trace
        main_trace.update({
            "metadata": {
                **(main_trace.metadata or {}),
                "error": {
                    "name": "ProcessingError",
                    "message": str(e),
                    "stack": getattr(e, "__traceback__", None)
                }
            }
        })
        
        print(f"Error processing user content: {e}")
    finally:
        # Complete the main trace
        main_trace.end()
    
    print("Test completed successfully!")
    return main_trace

if __name__ == "__main__":
    # Run the test
    test_monitor_sdk() 