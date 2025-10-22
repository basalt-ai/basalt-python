from typing import Dict, Optional, cast, Any
from ..ressources.monitor.base_log_types import BaseLogParamsWithType, LogType
from ..ressources.monitor.log_types import LogParams, Input, Output
from ..ressources.monitor.generation_types import GenerationParams
from .base_log import BaseLog
from .generation import Generation

class Log(BaseLog):
    """
    Class representing a log in the monitoring system.
    """
    def __init__(self, params: LogParams):
        base_log_params = BaseLogParamsWithType(
            name=params.get("name"),
            ideal_output=params.get("ideal_output"),
            start_time=params.get("start_time"),
            end_time=params.get("end_time"),
            metadata=params.get("metadata"),
            parent=params.get("parent"),
            trace=params.get("trace"),
            evaluators=params.get("evaluators"),
            type=LogType(params.get("type")),
        )

        super().__init__(base_log_params)
        self._input = params.get("input")
        self._output = None

    @property
    def input(self) -> Optional['Input']:
        """Get the log input."""
        return self._input

    @property
    def output(self) -> Optional['Output']:
        """Get the log output."""
        return self._output

    def start(self, input: Optional['Input'] = None) -> 'Log':
        """
        Start the log with an optional input.

        Args:
            input (Optional[str]): The input to the log.

        Returns:
            Log: The log instance.
        """
        if input:
            self._input = input

        super().start()
        return self

    def end(self, output: Optional['Output'] = None) -> 'Log':
        """
        End the log with an optional output.

        Args:
            output (Optional[str]): The output of the log.

        Returns:
            Log: The log instance.
        """
        super().end()

        if output:
            self._output = output

        return self

    def append(self, generation: 'Generation') -> 'Log':
        """
        Append a generation to this log.

        Args:
            generation (Generation): The generation to append.

        Returns:
            Log: The log instance.
        """
        # Remove child log from the list of its previous trace
        if generation.trace:
            generation.trace.logs = [log for log in generation.trace.logs if log.id != generation.id]

        # Add child to the new trace list
        self.trace.logs.append(cast(BaseLog, generation))

        # Set the trace of the generation to the current log
        generation.trace = self.trace
        generation.options = {"type": "multi"}

        # Set the parent of the generation to the current log
        generation.parent = self

        return self

    def update(self, params: Dict[str, Any]) -> 'Log':
        """
        Update the log with new parameters.

        Args:
            params (Dict[str, Any]): Parameters to update.

        Returns:
            Log: The log instance.
        """
        super().update(params)

        if "output" in params:
            self._output = params["output"]

        if "input" in params:
            self._input = params["input"]

        return self

    def create_generation(self, params: GenerationParams) -> 'Generation':
        """
        Create a new generation as a child of this log.

        Args:
            params (GenerationParams): Parameters for the generation.

        Returns:
            Generation: The new generation instance.
        """
        # Set the name to the prompt slug only if name is not provided
        name = params.get("name")
        if not name and params.get("prompt") and params["prompt"].get("slug"):
            name = params["prompt"]["slug"]

        # Create a new params dict to avoid modifying the original
        generation_params_dict = {**params, "name": name, "trace": self.trace, "parent": self}
        generation_params = GenerationParams(**generation_params_dict)
        generation = Generation(generation_params)

        return generation

    def create_log(self, params: LogParams) -> 'Log':
        """
        Create a new log as a child of this log.

        Args:
            params (LogParams): Parameters for the log.

        Returns:
            Log: The new log instance.
        """
        log_params = LogParams(**params, trace=self.trace, parent=self)
        log = Log(log_params)

        return log
