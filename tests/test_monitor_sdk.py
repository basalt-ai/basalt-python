import unittest
from unittest.mock import MagicMock

from basalt.sdk.monitorsdk import MonitorSDK
from basalt.sdk.promptsdk import PromptSDK
from basalt.utils.logger import Logger
from basalt.utils.dtos import PromptResponse, PromptModel
from basalt.endpoints.get_prompt import GetPromptEndpointResponse
from basalt.objects.trace import Trace
from basalt.objects.generation import Generation
from basalt.objects.log import Log

# Mock classes for testing
class MockOpenAI:
    """Mock OpenAI client for demonstration purposes."""
    
    def generate_text(self, prompt: str) -> str:
        """Generate text using a mock OpenAI."""
        return f"Generated response for: {prompt[:50]}..."
    
    def classify_content(self, content: str) -> str:
        """Classify content using a mock OpenAI."""
        return "Classification: Technology, Healthcare, AI"
    
    def translate_text(self, text: str) -> str:
        """Translate text using a mock OpenAI."""
        return "Traducción: Este es un texto traducido al español."
    
    def summarize_text(self, text: str) -> str:
        """Summarize text using a mock OpenAI."""
        return "Summary: This is a concise summary of the provided content."

# Setup common test objects
logger = Logger()
mocked_api = MagicMock()
mocked_api.invoke.return_value = (None, None)  # Default return value

class TestMonitorSDK(unittest.TestCase):
    """Test cases for the MonitorSDK class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.monitor = MonitorSDK(mocked_api, logger)
        self.openai = MockOpenAI()
        
        # Common test data
        self.user = {"id": "user123", "name": "John Doe"}
        self.content = "Create a technical article about machine learning applications in healthcare"
    
    def test_create_trace(self):
        """Test creating a trace."""
        trace = self.monitor.create_trace(
            "test-slug",
            {
                "input": self.content,
                "user": self.user,
                "organization": {"id": "org-123", "name": "Basalt"},
                "metadata": {"property1": "value1", "property2": "value2"},
                "name": "Test Trace"
            }
        )
        
        # Assert trace was created correctly
        self.assertIsNotNone(trace)
        self.assertIsInstance(trace, Trace)
        self.assertEqual(trace.input, self.content)
        self.assertEqual(trace.user, self.user)
        self.assertEqual(trace.organization, {"id": "org-123", "name": "Basalt"})
        self.assertEqual(trace.metadata, {"property1": "value1", "property2": "value2"})
        self.assertEqual(trace.chain_slug, "test-slug")
    
    def test_create_log(self):
        """Test creating a log within a trace."""
        trace = self.monitor.create_trace("test-slug", {"input": self.content})
        
        log = trace.create_log({
            "type": "span",
            "name": "test-log",
            "input": self.content,
            "metadata": {"property1": "value1"}
        })
        
        # Assert log was created correctly
        self.assertIsNotNone(log)
        self.assertIsInstance(log, Log)
        self.assertEqual(log.input, self.content)
        self.assertEqual(log.name, "test-log")
        self.assertEqual(log.metadata, {"property1": "value1"})
        self.assertEqual(log.trace, trace)
        
        # Assert log is in trace logs
        self.assertIn(log, trace.logs)
    
    def test_create_generation(self):
        """Test creating a generation within a log."""
        trace = self.monitor.create_trace("test-slug", {"input": self.content})
        
        log = trace.create_log({
            "type": "span",
            "name": "test-log",
            "input": self.content
        })
        
        generation = log.create_generation({
            "name": "test-generation",
            "input": self.content,
            "prompt": {"slug": "test-prompt", "version": "1.0"},
            "variables": [{"label": "var1", "value": "value1"}]
        })
        
        # Assert generation was created correctly
        self.assertIsNotNone(generation)
        self.assertIsInstance(generation, Generation)
        self.assertEqual(generation.input, self.content)
        self.assertEqual(generation.name, "test-generation")
        self.assertEqual(generation.prompt, {"slug": "test-prompt", "version": "1.0"})
        self.assertEqual(generation.variables, [{"label": "var1", "value": "value1"}])
        self.assertEqual(generation.trace, trace)
    
    def test_update_log(self):
        """Test updating a log."""
        trace = self.monitor.create_trace("test-slug", {"input": self.content})
        
        log = trace.create_log({
            "type": "span",
            "name": "test-log",
            "input": self.content
        })
        
        # Update the log
        log.update({
            "metadata": {"updated": True, "timestamp": "2023-01-01"},
            "output": "Updated output"
        })
        
        # Assert log was updated correctly
        self.assertEqual(log.output, "Updated output")
        self.assertEqual(log.metadata, {"updated": True, "timestamp": "2023-01-01"})
    
    def test_update_generation(self):
        """Test updating a generation."""
        trace = self.monitor.create_trace("test-slug", {"input": self.content})
        
        log = trace.create_log({
            "type": "span",
            "name": "test-log",
            "input": self.content
        })
        
        generation = log.create_generation({
            "name": "test-generation",
            "input": self.content
        })
        
        # Update the generation
        generation.update({
            "metadata": {"updated": True, "timestamp": "2023-01-01"},
            "output": "Updated output",
            "prompt": {"slug": "updated-prompt"}
        })
        
        # Assert generation was updated correctly
        self.assertEqual(generation.output, "Updated output")
        self.assertEqual(generation.metadata, {"updated": True, "timestamp": "2023-01-01"})
        self.assertEqual(generation.prompt, {"slug": "updated-prompt"})
    
    def test_end_log(self):
        """Test ending a log."""
        trace = self.monitor.create_trace("test-slug", {"input": self.content})
        
        log = trace.create_log({
            "type": "span",
            "name": "test-log",
            "input": self.content
        })
        
        # End the log
        log.end("Log output")
        
        # Assert log was ended correctly
        self.assertEqual(log.output, "Log output")
        self.assertIsNotNone(log.end_time)
    
    def test_end_generation(self):
        """Test ending a generation."""
        trace = self.monitor.create_trace("test-slug", {"input": self.content})
        
        log = trace.create_log({
            "type": "span",
            "name": "test-log",
            "input": self.content
        })
        
        generation = log.create_generation({
            "name": "test-generation",
            "input": self.content
        })
        
        # End the generation
        generation.end("Generation output")
        
        # Assert generation was ended correctly
        self.assertEqual(generation.output, "Generation output")
        self.assertIsNotNone(generation.end_time)
    
    def test_end_trace(self):
        """Test ending a trace."""
        trace = self.monitor.create_trace("test-slug", {"input": self.content})
        
        # End the trace
        trace.end("Trace output")
        
        # Assert trace was ended correctly
        self.assertEqual(trace.output, "Trace output")
        self.assertIsNotNone(trace.end_time)
    
    def test_nested_logs(self):
        """Test creating nested logs."""
        trace = self.monitor.create_trace("test-slug", {"input": self.content})
        
        parent_log = trace.create_log({
            "type": "span",
            "name": "parent-log",
            "input": self.content
        })
        
        child_log = parent_log.create_log({
            "type": "span",
            "name": "child-log",
            "input": "Child input"
        })
        
        # Assert parent-child relationship
        self.assertEqual(child_log.parent, parent_log)
        self.assertEqual(child_log.trace, trace)
    
    def test_trace_identify(self):
        """Test identifying a trace with user and organization."""
        trace = self.monitor.create_trace("test-slug", {"input": self.content})
        
        # Identify the trace
        trace.identify({
            "user": self.user,
            "organization": {"id": "org-123", "name": "Basalt"}
        })
        
        # Assert trace was identified correctly
        self.assertEqual(trace.user, self.user)
        self.assertEqual(trace.organization, {"id": "org-123", "name": "Basalt"})


class TestMonitorSDKIntegration(unittest.TestCase):
    """Integration tests for the MonitorSDK with PromptSDK."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.monitor = MonitorSDK(mocked_api, logger)
        self.openai = MockOpenAI()
        
        # Common test data
        self.user = {"id": "user456", "name": "Jane Smith"}
        self.query = "What are the best practices for machine learning model deployment?"
    
    def test_prompt_generation_integration(self):
        """Test the integration between PromptSDK and MonitorSDK."""
        # Create a main trace
        main_trace = self.monitor.create_trace(
            "prompt-generation-test",
            {
                "input": self.query,
                "user": self.user,
                "organization": {"id": "org-456", "name": "Basalt Testing"},
                "metadata": {"source": "test", "environment": "development"},
                "name": "Prompt Generation Test"
            }
        )
        
        # Create a span for the prompt generation
        prompt_span = main_trace.create_log({
            "type": "span",
            "name": "prompt-retrieval",
            "input": self.query,
            "metadata": {"action": "retrieve-prompt"}
        })
        
        # Mock the API response for get_prompt
        mock_prompt_response = GetPromptEndpointResponse(
            warning=None,
            prompt=PromptResponse(
                text="Answer the following question about {{topic}}: {{question}}",
                model=PromptModel(
                    provider="open-ai",
                    model="gpt-4o",
                    version="latest",
                    parameters={
                        "temperature": 0.7,
                        "topP": 1,
                        "maxLength": 4096,
                        "responseFormat": "text"
                    }
                )
            )
        )
        
        # Create a mock API for PromptSDK
        prompt_api = MagicMock()
        prompt_api.invoke.return_value = (None, mock_prompt_response)
        
        # Create a PromptSDK instance
        from basalt.utils.memcache import MemoryCache
        
        # Create a PromptSDK instance
        prompt_sdk = PromptSDK(
            api=prompt_api,
            cache=MemoryCache(),
            fallback_cache=MemoryCache(),
            logger=logger
        )
        
        # Get prompt from Basalt
        err, prompt_response, generation = prompt_sdk.get(
            "ml-best-practices", 
            variables={"topic": "machine learning", "question": self.query},
            version="1.0"
        )
        
        # Verify the prompt was retrieved successfully
        self.assertIsNone(err)
        self.assertIsNotNone(prompt_response)
        self.assertIsNotNone(generation)
        
        # Verify prompt response properties
        expected_text = "Answer the following question about machine learning: What are the best practices for machine learning model deployment?"
        self.assertEqual(prompt_response.text, expected_text)
        self.assertEqual(prompt_response.model.provider, "open-ai")
        self.assertEqual(prompt_response.model.model, "gpt-4o")
        
        # Verify generation object properties
        self.assertEqual(generation.prompt["slug"], "ml-best-practices")
        self.assertEqual(generation.prompt["version"], "1.0")
        self.assertEqual(generation.input, "Answer the following question about {{topic}}: {{question}}")
        self.assertEqual(generation.variables, [
            {"label": "topic", "value": "machine learning"}, 
            {"label": "question", "value": self.query}
        ])
        self.assertEqual(generation.options["type"], "single")
        
        # End the prompt span
        prompt_span.end(prompt_response.text)
        
        # Create a span for the model generation
        model_span = main_trace.create_log({
            "type": "span",
            "name": "model-generation",
            "input": prompt_response.text,
            "metadata": {"action": "generate-response"}
        })
        
        # Generate text using OpenAI
        model_response = self.openai.generate_text(prompt_response.text)
        
        # Update the generation with the output
        generation.end(model_response)
        
        # End the model span
        model_span.end(model_response)
        
        # End the main trace
        main_trace.end("Completed prompt generation test")
        
        # Verify trace structure
        # Filter logs to only include those of type "span"
        span_logs = [log for log in main_trace.logs if log.type == "span"]
        self.assertEqual(len(span_logs), 2)
        self.assertEqual(span_logs[0], prompt_span)
        self.assertEqual(span_logs[1], model_span)
    
    def test_complex_workflow(self):
        """Test a complex workflow with multiple generations and spans."""
        # Create a main trace
        main_trace = self.monitor.create_trace(
            "complex-workflow-test",
            {
                "input": self.query,
                "user": self.user,
                "name": "Complex Workflow Test"
            }
        )
        
        # Step 1: Content generation
        generation_span = main_trace.create_log({
            "type": "span",
            "name": "content-generation",
            "input": self.query
        })
        
        # Mock the API response for get_prompt
        mock_generate_prompt_response = GetPromptEndpointResponse(
            warning=None,
            prompt=PromptResponse(
                text="Generate content about: {{query}}",
                model=PromptModel(
                    provider="open-ai",
                    model="gpt-4o",
                    version="latest",
                    parameters={
                        "temperature": 0.7,
                        "topP": 1,
                        "maxLength": 4096,
                        "responseFormat": "text"
                    }
                )
            )
        )
        
        # Create a mock API for PromptSDK
        prompt_api = MagicMock()
        prompt_api.invoke.return_value = (None, mock_generate_prompt_response)
        
        # Create a PromptSDK instance
        from basalt.utils.memcache import MemoryCache
        prompt_sdk = PromptSDK(
            api=prompt_api,
            cache=MemoryCache(),
            fallback_cache=MemoryCache(),
            logger=logger
        )
        
        # Get prompt from Basalt
        err, prompt_response, generation = prompt_sdk.get(
            "generate-content", 
            variables={"query": self.query},
            version="1.0"
        )
        
        # Create generation log
        generation_log = generation_span.create_generation({
            "name": "text-generation",
            "input": self.query,
            "prompt": {"slug": "generate-content", "version": "1.0"},
            "variables": [{"label": "query", "value": self.query}]
        })
        
        # Generate text
        generated_text = self.openai.generate_text(prompt_response.text)
        
        # Update generation with output
        generation_log.update({
            "output": generated_text,
            "metadata": {"processingTime": 500}
        })
        
        generation_span.end(generated_text)
        
        # Step 2: Classification
        classification_span = main_trace.create_log({
            "type": "span",
            "name": "classification",
            "input": generated_text
        })
        
        # Mock the API response for classification prompt
        mock_classify_prompt_response = GetPromptEndpointResponse(
            warning=None,
            prompt=PromptResponse(
                text="Classify the following content: {{content}}",
                model=PromptModel(
                    provider="open-ai",
                    model="gpt-4o",
                    version="latest",
                    parameters={
                        "temperature": 0.3,
                        "topP": 1,
                        "maxLength": 2048,
                        "responseFormat": "text"
                    }
                )
            )
        )
        
        # Update the mock API for the classification prompt
        prompt_api.invoke.return_value = (None, mock_classify_prompt_response)
        
        # Get prompt from Basalt
        err, classify_prompt_response, classify_generation = prompt_sdk.get(
            "classify-content",
            variables={"content": generated_text},
            version="1.0"
        )
        
        # Create generation log
        class_gen = classification_span.create_generation({
            "name": "content-classification",
            "input": generated_text,
            "prompt": {"slug": "classify-content", "version": "1.0"},
            "variables": [{"label": "content", "value": generated_text}]
        })
        
        # Classify content
        categories = self.openai.classify_content(generated_text)
        
        # Update generation with output
        class_gen.update({
            "output": categories
        })
        
        classification_span.end(categories)
        
        # End the main trace
        main_trace.end("Workflow completed")
        
        # Verify trace structure
        # Filter logs to only include those of type "span"
        span_logs = [log for log in main_trace.logs if log.type == "span"]
        self.assertEqual(len(span_logs), 2)
        self.assertEqual(span_logs[0], generation_span)
        self.assertEqual(span_logs[1], classification_span)
        
        # Verify outputs
        self.assertEqual(generation_span.output, generated_text)
        self.assertEqual(classification_span.output, categories)
        self.assertEqual(main_trace.output, "Workflow completed")


if __name__ == "__main__":
    unittest.main() 