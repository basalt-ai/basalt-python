# %% [markdown]
# # Basalt Monitor SDK Demo
# 
# This notebook demonstrates how to use the Basalt Monitor SDK to track and monitor your AI application's execution.

# %%
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

os.environ["BASALT_BUILD"] = "development"

from basalt import Basalt

# Initialize the SDK
basalt = Basalt(
	api_key="sk-d4ef0c22e36364fa911119aa04d52948f597304e32257a9901005998bd3bfe4b", # Replace with your API key
	log_level="debug" # Optional: Set log level
)  

# %% [markdown]
# ## 1. Creating a Basic Trace
# 
# A trace represents a complete execution flow in your application. Let's create a simple trace:

# %%
# Create a trace
trace = basalt.monitor.create_trace(
    "theo-slug",  # Chain slug - identifies this type of workflow
    {
        "input": "What are the benefits of AI in healthcare?",
        "user": {"id": "user123", "name": "John Doe"},
        "organization": {"id": "org123", "name": "Healthcare Inc"},
        "metadata": {"source": "web", "priority": "high"}
    }
)

print(f"Created trace with input: {trace.input}")

# %% [markdown]
# ## 2. Adding Logs to a Trace
# 
# Logs represent individual steps or operations within a trace:

# %%
# Create a log for content moderation
moderation_log = trace.create_log({
    "type": "span",
    "name": "content-moderation",
    "input": trace.input,
    "metadata": {"model": "text-moderation-latest"}
})

# Simulate moderation check
moderation_result = {"flagged": False, "categories": [], "scores": {}}

# Update and end the log
moderation_log.update({"metadata": {"completed": True}})
moderation_log.end(moderation_result)

print(f"Completed moderation check: {moderation_log.output}")

# %% [markdown]
# ## 3. Creating and Managing Generations
# 
# Generations are special types of logs specifically for AI model interactions:

# %%
# Create a log for the main processing
main_log = trace.create_log({
    "type": "span",
    "name": "main-processing",
    "input": trace.input
})

# Create a generation within the main log
generation = main_log.create_generation({
    "name": "healthcare-benefits-generation",
    "input": trace.input,
    "prompt": {
        "slug": "healthcare-benefits",
        "version": "1.0"
    }
})

# Simulate AI response
ai_response = """
AI in healthcare offers numerous benefits:
1. Early disease detection through advanced imaging analysis
2. Personalized treatment recommendations
3. Automated administrative tasks
4. Enhanced drug discovery process
5. Improved patient monitoring
"""

# End the generation with the response
generation.end(ai_response)

# End the main log
main_log.end(ai_response)

trace.end("End of trace")

print(f"Generated response:\n{generation.output}")

# %% [markdown]
# ## 4. Complex Workflow Example
# 
# Here's a more complex example showing nested logs and multiple generations:

# %%
# Create a new trace for a complex workflow
complex_trace = basalt.monitor.create_trace(
    "medical-report-analysis",
    {
        "input": "Patient presents with frequent headaches and fatigue.",
        "metadata": {"department": "neurology", "priority": "high"}
    }
)

# Initial analysis log
analysis_log = complex_trace.create_log({
    "type": "span",
    "name": "symptom-analysis",
    "input": complex_trace.input
})

# Generate initial analysis
analysis_gen = analysis_log.create_generation({
    "name": "symptom-classification",
    "input": complex_trace.input,
    "prompt": {"slug": "classify-symptoms", "version": "1.0"}
})
analysis_gen.end("Primary symptoms suggest possible migraine or chronic fatigue syndrome")

# Create a nested log for recommendations
recommendations_log = analysis_log.create_log({
    "type": "span",
    "name": "treatment-recommendations",
    "input": analysis_gen.output
})

# Generate treatment recommendations
treatment_gen = recommendations_log.create_generation({
    "name": "treatment-suggestions",
    "input": analysis_gen.output,
    "prompt": {"slug": "suggest-treatments", "version": "1.0"}
})

treatment_response = """
Recommended treatments:
1. Schedule neurological examination
2. Keep headache diary for pattern recognition
3. Consider sleep study for fatigue assessment
4. Initial blood work to rule out underlying conditions
"""
treatment_gen.end(treatment_response)

# End all logs
recommendations_log.end(treatment_response)

# Convert object to string for analysis_log
analysis_summary = f"Analysis: {analysis_gen.output}\nRecommendations: {treatment_response}"
analysis_log.end(analysis_summary)

# End the complex trace
complex_trace.end("Medical report analysis completed")

print("Completed medical report analysis workflow")
print(f"Analysis: {analysis_gen.output}")
print(f"Recommendations: {treatment_gen.output}")
