"""Example demonstrating the new observe decorator API."""

from basalt.observability import (
    ObserveKind,
    observe,
    start_observe,
)


@observe(kind=ObserveKind.SPAN, name="process.data")
def process_data(text: str) -> str:
    """Process some data."""
    return text.upper()


@observe(kind=ObserveKind.GENERATION, name="llm.generate")
def generate_text(prompt: str, model: str = "gpt-4") -> str:
    """Generate text with an LLM."""
    # Simulated LLM call
    return f"Generated response for: {prompt}"


@observe( name="vector.search", kind=ObserveKind.RETRIEVAL)
def search_documents(query: str) -> list[dict]:
    """Search documents in vector database."""
    return [
        {"id": 1, "content": "Document 1", "score": 0.95},
        {"id": 2, "content": "Document 2", "score": 0.87},
    ]


@observe(kind=ObserveKind.SPAN, name="workflow.execute")
def execute_workflow(steps: list[str]) -> dict:
    """Execute a workflow with multiple steps."""
    return {"status": "completed", "steps_executed": len(steps)}


@observe(
    name="llm.chat",
    kind=ObserveKind.GENERATION,
    evaluators=["quality", "relevance"],
)
def chat_with_llm(message: str) -> str:
    """Chat with an LLM with quality evaluation."""
    return f"Response to: {message}"




@start_observe(
    name="main_workflow",
    identity={
        "organization": {"id": "123", "name": "Demo Corp"},
        "user": {"id": "456", "name": "Alice"}
    },
    experiment={"id": "exp_123"},
    metadata={"environment": "demo"},
)
def main():
    """Main workflow demonstrating nested observe calls."""
    # Test the new decorators
    observe.input({"workflow": "demo_suite"})

    result1 = process_data("hello world")

    result2 = generate_text("Write a poem")

    result3 = search_documents("machine learning")

    result4 = execute_workflow(["step1", "step2", "step3"])

    result5 = chat_with_llm("How are you?")

    observe.output({"results_count": 5})
    return {"completed": True}


if __name__ == "__main__":
    main()



