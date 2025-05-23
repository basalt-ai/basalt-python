from datetime import datetime
from typing import Dict, Optional, Any, List


from ..ressources.monitor.trace_types import TraceParams
from .base_log import BaseLog
from .generation import Generation
from ..utils.flusher import Flusher
from .experiment import Experiment
from ..ressources.monitor.evaluator_types import Evaluator
from ..utils.logger import Logger

class Trace:
    """
    Class representing a trace in the monitoring system.
    """
    def __init__(self, feature_slug: str, params: TraceParams, flusher: 'Flusher', logger: 'Logger'):
        self._feature_slug = feature_slug

        self._input = params.get("input")
        self._output = params.get("output")
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
        self._evaluation_config = params.get("evaluationConfig")
        self._logger = logger

        self._experiment = None

        if "experiment" in params:
            experiment = params["experiment"]
            if experiment is None:
                self._logger.warn("Warning: Experiment is None. This experiment will be ignored.")
            elif experiment.feature_slug != self._feature_slug:
                self._logger.warn("Warning: Experiment feature slug does not match trace feature slug. This experiment will be ignored.")
            else:
                self._experiment = experiment

    @property
    def name(self) -> Optional[str]:
        """Get the trace name."""
        return self._name

    @property
    def input(self) -> Optional[str]:
        """Get the trace input."""
        return self._input

    @property
    def output(self) -> Optional[str]:
        """Get the trace output."""
        return self._output

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

    def start(self, input: Optional[str] = None) -> 'Trace':
        """
        Start the trace with an optional input.

        Args:
            input (Optional[str]): The input to the trace.

        Returns:
            Trace: The trace instance.
        """
        if input:
            self._input = input

        self._start_time = datetime.now()
        return self

    def identify(self, params: Dict[str, Any]) -> 'Trace':
        """
        Set identification information for the trace.

        Args:
            params (Dict[str, Any]): Identification parameters.

        Returns:
            Trace: The trace instance.
        """
        self._user = params.get("user")
        self._organization = params.get("organization")
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

    def set_evaluation_config(self, config: Dict[str, Any]) -> 'Trace':
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

    def update(self, params: Dict[str, Any]) -> 'Trace':
        """
        Update the trace.

        Args:
            params (Dict[str, Any]): Parameters to update.

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
        self._evaluation_config = params.get("evaluationConfig", self._evaluation_config)

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
        generation.trace = self
        
        return self

    def create_generation(self, params: Dict[str, Any]) -> 'Generation':
        """
        Create a new generation in this trace.
        
        Args:
            params (Dict[str, Any]): Parameters for the generation.
            
        Returns:
            Generation: The new generation instance.
        """
        from .generation import Generation
        
        # Set the name to the prompt slug if available
        name = params.get("name")
        if params.get("prompt") and params["prompt"].get("slug"):
            name = params["prompt"]["slug"]
            
        generation = Generation({
            **params,
            "name": name,
            "trace": self
        })
        
        return generation

    def create_log(self, params: Dict[str, Any]) -> 'BaseLog':
        """
        Create a new log in this trace.
        
        Args:
            params (Dict[str, Any]): Parameters for the log.
            
        Returns:
            Log: The new log instance.
        """
        from .log import Log
        
        log = Log({
            **params,
            "trace": self
        })

        return log

    def end(self, output: Optional[str] = None) -> 'Trace':
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
            self._flusher.flush_trace(self)

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert the trace to a dictionary for API serialization."""
        return {
            "feature_slug": self._feature_slug,
            "input": self._input,
            "output": self._output,
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