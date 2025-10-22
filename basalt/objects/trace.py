from datetime import datetime
from typing import Dict, Optional, Any, List

from ..ressources.monitor.trace_types import TraceParams, User, Organization, Input, Output, IdealOutput
from .base_log import BaseLog
from .generation import Generation
from .log import Log
from ..utils.flusher import Flusher
from .experiment import Experiment
from ..ressources.monitor.evaluator_types import Evaluator, EvaluationConfig
from ..ressources.monitor.generation_types import GenerationParams
from ..ressources.monitor.log_types import LogParams
from ..utils.protocols import ILogger

class Trace:
    """
    Class representing a trace in the monitoring system.
    """
    def __init__(self, feature_slug: str, params: TraceParams, flusher: 'Flusher', logger: 'ILogger'):
        self._feature_slug = feature_slug

        self._input = params.get('input')
        self._output = params.get("output")
        self._ideal_output = params.get("ideal_output")
        self._name = params.get("name")
        self._start_time = params.get("start_time", datetime.now())
        self._end_time = params.get("end_time")
        self._user = params.get("user")
        self._organization = params.get("organization")
        self._metadata = params.get("metadata")

        self._logs: List['BaseLog'] = []

        self._flusher = flusher
        self._is_ended = False

        self._evaluators = params.get("evaluators")
        self._evaluation_config = params.get("evaluation_config")
        self._logger = logger

        self._experiment = None

        if "experiment" in params:
            experiment = params["experiment"]
            if experiment.feature_slug != self._feature_slug:
                self._logger.warn("Warning: Experiment feature slug does not match trace feature slug. This experiment will be ignored.")
            else:
                self._experiment = experiment

    @property
    def name(self) -> Optional[str]:
        """Get the trace name."""
        return self._name

    @property
    def input(self) -> Optional[Input]:
        """Get the trace input."""
        return self._input

    @property
    def output(self) -> Optional[Output]:
        """Get the trace output."""
        return self._output

    @property
    def ideal_output(self) -> Optional[IdealOutput]:
        """Get the trace ideal_output."""
        return self._ideal_output

    @property
    def start_time(self) -> datetime:
        """Get the start time."""
        return self._start_time

    @property
    def user(self) -> Optional[Dict[str, Any]]:
        """Get the user information."""
        return self._user

    @property
    def organization(self) -> Optional[Dict[str, Any]]:
        """Get the organization information."""
        return self._organization

    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        """Get the metadata."""
        return self._metadata

    @property
    def logs(self) -> List['BaseLog']:
        """Get the logs."""
        return self._logs

    @logs.setter
    def logs(self, logs: List['BaseLog']):
        """Set the logs."""
        self._logs = logs

    @property
    def feature_slug(self) -> str:
        """Get the feature slug."""
        return self._feature_slug

    @property
    def end_time(self) -> Optional[datetime]:
        """Get the end time."""
        return self._end_time

    @property
    def experiment(self) -> Optional['Experiment']:
        """Get the experiment."""
        return self._experiment

    @property
    def evaluation_config(self) -> Optional[Dict[str, Any]]:
        """Get the evaluation configuration."""
        return self._evaluation_config

    @property
    def evaluators(self) -> Optional[List[Dict[str, Any]]]:
        """Get the evaluators."""
        return self._evaluators

    def start(self, input: Optional[Input] = None) -> 'Trace':
        """
        Start the trace with an optional input.

        Args:
            input (Optional[Input]): The input to the trace. Can be a string, an array or a dictionary.

        Returns:
            Trace: The trace instance.
        """
        if input:
            self._input = input

        self._start_time = datetime.now()
        return self

    def set_ideal_output(self, ideal_output: str) -> 'Trace':
        """Sets the ideal output for the trace."""
        self._ideal_output = ideal_output
        return self

    def identify(self, user: User = {}, organization: Organization = {}) -> 'Trace':
        """
        Set identification information for the trace.

        Args:
            user: The user information to associate with this trace.
            organization: The organization information to associate with this trace.

        Returns:
            Trace: The trace instance.
        """
        self._user = user
        self._organization = organization
        return self

    def set_metadata(self, metadata: Dict[str, Any]) -> 'Trace':
        """
        Set metadata for the trace.

        Args:
            metadata (Dict[str, Any]): The metadata to set.

        Returns:
            Trace: The trace instance.
        """
        self._metadata = metadata
        return self

    def set_evaluation_config(self, config: EvaluationConfig) -> 'Trace':
        """
        Set the evaluation configuration for the trace.

        Args:
            config (Dict[str, Any]): The evaluation configuration to set.

        Returns:
            Trace: The trace instance.
        """
        self._evaluation_config = config
        return self

    def set_experiment(self, experiment: Dict[str, Any]) -> 'Trace':
        """
        Set the experiment for the trace.

        Args:
            experiment (Dict[str, Any]): The experiment to set.

        Returns:
            Trace: The trace instance.
        """
        self._experiment = experiment
        return self

    def add_evaluator(self, evaluator: Evaluator) -> 'Trace':
        """
        Add an evaluator to the trace.

        Args:
            evaluator (Dict[str, Any]): The evaluator to add.

        Returns:
            Trace: The trace instance.
        """
        if self._evaluators is None:
            self._evaluators = []

        self._evaluators.append(evaluator)
        return self

    def update(self, params: TraceParams) -> 'Trace':
        """
        Update the trace.

        Args:
            params (TraceParams): Parameters to update.

        Returns:
            Trace: The trace instance.
        """
        self._metadata = params.get("metadata", self._metadata)
        self._input = params.get("input", self._input)
        self._output = params.get("output", self._output)
        self._organization = params.get("organization", self._organization)
        self._user = params.get("user", self._user)

        if params.get("start_time"):
            self._start_time = params.get("start_time")

        if params.get("end_time"):
            self._end_time = params.get("end_time")

        self._name = params.get("name", self._name)
        self._evaluators = params.get("evaluators", self._evaluators)
        self._evaluation_config = params.get("evaluation_config", self._evaluation_config)

        return self

    def append(self, generation: 'Generation') -> 'Trace':
        """
        Append a generation to this trace.

        Args:
            generation (Generation): The generation to append.

        Returns:
            Trace: The trace instance.
        """
        # Remove child log from the list of its previous trace
        if generation.trace:
            generation.trace.logs = [log for log in generation.trace.logs if log.id != generation.id]

        # Add child to the new trace list
        self._logs.append(generation)

        # Set the trace of the generation to the current log
        generation.trace = self
        generation.options = {"type": "multi"}

        return self

    def create_generation(self, params: GenerationParams) -> 'Generation':
        """
        Create a new generation in this trace.

        Args:
            params (GenerationParams): Parameters for the generation.

        Returns:
            Generation: The new generation instance.
        """
        generation_params = GenerationParams(**params, trace=self)
        generation = Generation(generation_params)

        return generation

    def create_log(self, params: LogParams) -> 'BaseLog':
        """
        Create a new log in this trace.

        Args:
            params (LogParams): Parameters for the log.

        Returns:
            Log: The new log instance.
        """
        log_params = LogParams(**params, trace=self)
        log = Log(log_params)

        return log

    async def end(self, output: Optional[Output] = None) -> 'Trace':
        """
        End the trace with an optional output.

        Args:
            output (Optional[str]): The output of the trace.

        Returns:
            Trace: The trace instance.
        """
        self._output = output if output is not None else self._output

        # Send to the API using the flusher
        if self._can_flush():
            self._end_time = datetime.now()
            self._is_ended = True
            await self._flusher.flush_trace(self)

        return self

    def end_sync(self, output: Optional[Output] = None) -> 'Trace':
        """
        End the trace with an optional output synchronously.

        Args:
            output (Optional[str]): The output of the trace.

        Returns:
            Trace: The trace instance.
        """
        self._output = output if output is not None else self._output

        # Send to the API using the flusher
        if self._can_flush():
            self._end_time = datetime.now()
            self._is_ended = True
            self._flusher.flush_trace_sync(self)

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert the trace to a dictionary for API serialization."""
        return {
            "feature_slug": self._feature_slug,
            "input": self._input,
            "output": self._output,
            "ideal_output": self._ideal_output,
            "name": self._name,
            "start_time": self._start_time,
            "end_time": self._end_time,
            "user": self._user,
            "organization": self._organization,
            "metadata": self._metadata,
            "logs": self._logs,
            "experiment": self._experiment,
            "evaluators": self._evaluators,
            "evaluation_config": self._evaluation_config
        }

    def _can_flush(self) -> bool:
        """
        Check if the trace can be flushed.

        Returns:
            bool: True if the trace can be flushed, False otherwise.
        """
        if self._is_ended:
            self._logger.warn('Trace already ended. This operation will be ignored.')

        return not self._is_ended
