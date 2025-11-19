"""Example demonstrating the new observe decorator API."""

from basalt.observability import (
    ObserveKind,
    observe,
)


# Example 1: Using observe with kind parameter
@observe(ObserveKind.SPAN, name="process.data")
def process_data(text: str) -> str:
    """Process some data."""
    return text.upper()


# Example 2: Using observe_generation (specialized decorator)
@observe(kind=ObserveKind.GENERATION, name="llm.generate")
def generate_text(prompt: str, model: str = "gpt-4") -> str:
    """Generate text with an LLM."""
    # Simulated LLM call
    return f"Generated response for: {prompt}"


# Example 3: Using observe with string kind
@observe("retrieval", name="vector.search")
def search_documents(query: str) -> list[dict]:
    """Search documents in vector database."""
    return [
        {"id": 1, "content": "Document 1", "score": 0.95},
        {"id": 2, "content": "Document 2", "score": 0.87},
    ]


# Example 4: Using observe_span for general operations
@observe(kind=ObserveKind.SPAN, name="workflow.execute")
def execute_workflow(steps: list[str]) -> dict:
    """Execute a workflow with multiple steps."""
    return {"status": "completed", "steps_executed": len(steps)}


# Example 5: Using observe with evaluators
@observe(
    ObserveKind.GENERATION,
    name="llm.chat",
    evaluators=["quality", "relevance"],
)
def chat_with_llm(message: str) -> str:
    """Chat with an LLM with quality evaluation."""
    return f"Response to: {message}"





if __name__ == "__main__":
    # Test the new decorators

    result1 = process_data("hello world")

    result2 = generate_text("Write a poem")

    result3 = search_documents("machine learning")

    result4 = execute_workflow(["step1", "step2", "step3"])

    result5 = chat_with_llm("How are you?")



