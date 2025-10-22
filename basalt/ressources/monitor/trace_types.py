from datetime import datetime
from typing import Dict, Optional, Any, List, TYPE_CHECKING, TypedDict
from dataclasses import dataclass, field
from .experiment_types import Experiment
from .evaluator_types import Evaluator, EvaluationConfig


if TYPE_CHECKING:
    from .generation_types import Generation, GenerationParams
    from .log_types import Log, LogParams
    from .base_log_types import BaseLog

@dataclass
class User(TypedDict):
    """User information associated with a trace."""
    id: str
    name: str

@dataclass
class Organization(TypedDict):
    """Organization information associated with a trace."""
    id: str
    name: str

Input = str | List[Any] | Dict[str, Any]
Output = str | List[Any] | Dict[str, Any]
IdealOutput = str | List[Any] | Dict[str, Any]

class TraceParams(TypedDict, total=False):
    """Parameters for creating or updating a trace."""
    name: Optional[str]
    input: Optional[Input]
    output: Optional[Output]
    ideal_output: Optional[IdealOutput]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    user: Optional[User]
    organization: Optional['Organization']
    metadata: Optional[Dict[str, Any]]
    experiment: Optional[Experiment]
    evaluators: List[Evaluator]
    evaluation_config: Optional[EvaluationConfig]


@dataclass
class Trace:
    """A trace represents a complete user interaction or process flow and serves as the top-level container for all monitoring activities.

    A trace provides methods to create and manage spans and generations within the process flow.

    Example:
        ```python
        # Create a basic trace
        trace = monitor_sdk.create_trace('user-query')

        # Start the trace with input
        trace.start('What is the capital of France?')

        # Create a span within the trace
        processing_log = trace.create_log(
            name='query-processing',
            type='process'
        )

        # Create a generation within the span
        generation = processing_log.create_generation(
            name='answer-generation',
            prompt={'slug': 'qa-prompt', 'version': '1.0.0'},
            input='What is the capital of France?'
        )

        # End the generation with output
        generation.end('The capital of France is Paris.')

        # End the span
        processing_log.end()

        # End the trace with final output
        trace.end('Paris is the capital of France.')
        ```
    """
    name: Optional[str]
    input: Optional[Input]
    output: Optional[Output]
    ideal_output: Optional[IdealOutput]
    start_time: datetime
    end_time: Optional[datetime]
    user: Optional[User]
    organization: Optional[Organization]
    metadata: Optional[Dict[str, Any]]
    experiment: Optional['Experiment']
    evaluators: List[Evaluator]
    evaluation_config: Optional[EvaluationConfig]
    logs: List['BaseLog'] = field(default_factory=list)

    def start(self, input: Optional[Input] = None) -> 'Trace':
        """Marks the trace as started and sets the input if provided.

        Args:
            input: Optional input data to associate with the trace.

        Returns:
            The trace instance for method chaining.

        Example:
            ```python
            # Start a trace without input
            trace.start()

            # Start a trace with input
            trace.start('User query: What is the capital of France?')
            ```
        """
        ...

    def set_ideal_output(self, ideal_output: IdealOutput) -> 'Trace':
        """Sets the ideal output for the trace."""
        ...

    def set_metadata(self, metadata: Dict[str, Any]) -> 'Trace':
        """Sets or updates the metadata for this trace.

        Args:
            metadata: The metadata to associate with this trace.

        Returns:
            The trace instance for method chaining.

        Example:
            ```python
            # Add metadata to the trace
            trace.set_metadata({
                'user_id': 'user-123',
                'session_id': 'session-456',
                'source': 'web-app'
            })
            ```
        """
        ...

    def set_evaluation_config(self, config: EvaluationConfig) -> 'Trace':
        """Sets the evaluation configuration for the trace.

        Args:
            config: The evaluation configuration to set.

        Returns:
            The trace instance for method chaining.
        """
        ...

    def set_experiment(self, experiment: Experiment) -> 'Trace':
        """Sets the experiment for the trace.

        Args:
            experiment: The experiment to set.

        Returns:
            The trace instance for method chaining.
        """
        ...

    def update(self, params: TraceParams) -> 'Trace':
        """Updates the trace with new parameters.
        The new parameters given in this method will override the existing ones.

        Args:
            params: The parameters to update.

        Returns:
            The trace instance for method chaining.

        Example:
            ```python
            # Update trace parameters
            trace.update({
                'name': 'Updated trace name',
                'metadata': {'priority': 'high'}
            })
            ```
        """
        ...

    def add_evaluator(self, evaluator: Evaluator) -> 'Trace':
        """Adds an evaluator to the trace.

        Args:
            evaluator: The evaluator to add to the trace.

        Returns:
            The trace instance for method chaining.
        """
        ...

    def append(self, log: 'BaseLog') -> 'Trace':
        """Adds a log (span or generation) to this trace.

        Args:
            log: The log to add to this trace.

        Returns:
            The trace instance for method chaining.

        Example:
            ```python
            # Create a generation separately and append it to the trace
            generation = monitor_sdk.create_generation(
                name='external-generation',
                trace=another_trace
            )

            # Append the generation to this trace
            trace.append(generation)
            ```
        """
        ...

    def identify(self, user: User = {}, organization: Organization = {}) -> 'Trace':
        """Associates user information with this trace.

        Args:
            user: The user information to associate with this trace.
            organization: The organization information to associate with this trace.

        Returns:
            The trace instance for method chaining.

        Example:
            ```python
            # Identify a user with user and organization information
            trace.identify(
                user=User(
                    id='user-123',
                    name='John Doe'
                ),
                organization=Organization(
                    id='org-123',
                    name='Acme Corporation'
                )
            )
            ```
        """
        ...

    def create_generation(self, params: 'GenerationParams') -> 'Generation':
        """Creates a new generation within this trace.

        Args:
            params: Parameters for the generation.

        Returns:
            A new Generation instance associated with this trace.

        Example:
            ```python
            # Create a generation with a prompt reference
            generation = trace.create_generation({
                'name': 'answer-generation',
                'prompt': {'slug': 'qa-prompt', 'version': '2.1.0'},
                'input': 'What is the capital of France?',
                'variables': {'style': 'concise', 'language': 'en'},
                'metadata': {'model_version': 'gpt-4'}
            })

            # Create a generation without a prompt reference
            simple_generation = trace.create_generation({
                'name': 'text-completion',
                'input': 'Complete this sentence: The sky is',
                'output': 'The sky is blue and vast.'
            })
            ```
        """
        ...

    def create_log(self, params: 'LogParams') -> 'Log':
        """Creates a new span within this trace.

        Args:
            params: Parameters for the span.

        Returns:
            A new Log instance associated with this trace.

        Example:
            ```python
            # Create a basic span
            basic_log = trace.create_log({
                'name': 'data-fetching',
                'type': 'io'
            })

            # Create a detailed span
            detailed_log = trace.create_log({
                'name': 'user-validation',
                'input': 'user credentials',
                'metadata': {'validation_rules': ['...word-strength', 'email-format']}
            })
            ```
        """
        ...

    async def end(self, output: Optional[Output] = None) -> 'Trace':
        """Marks the trace as ended and sets the output if provided.

        Args:
            output: Optional output data to associate with the trace.

        Returns:
            The trace instance for method chaining.

        Example:
            ```python
            # End a trace without output
            trace.end()

            # End a trace with text output
            trace.end('The capital of France is Paris.')

            # End a trace with json output
            trace.end({ "response": "The capital of France is Paris." })
            ```
        """
        ...

    def end_sync(self, output: Optional[Output] = None) -> 'Trace':
        """Marks the trace as ended and sets the output if provided.

        Args:
            output: Optional output data to associate with the trace.

        Returns:
            The trace instance for method chaining.

        Example:
            ```python
            # End a trace without output
            trace.end_sync()

            # End a trace with output
            trace.end_sync('The capital of France is Paris.')

            trace.end_sync({ "response": 'The capital of France is Paris.' })
            ```
        """
        ...
