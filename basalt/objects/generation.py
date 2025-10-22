from typing import Dict, Optional, Any, List

from .base_log import BaseLog
from ..ressources.monitor.generation_types import GenerationParams
from ..ressources.monitor.base_log_types import LogType, Input, Output, BaseLogParamsWithType


class Generation(BaseLog):
    """
    Class representing a generation in the monitoring system.
    """
    def __init__(self, params: GenerationParams):
        base_log_params = BaseLogParamsWithType(
            name= params.get("name"),
            input=params.get("input"),
            output=params.get("output"),
            ideal_output=params.get("ideal_output"),
            start_time=params.get("start_time"),
            end_time=params.get("end_time"),
            metadata=params.get("metadata"),
            parent=params.get("parent"),
            trace=params.get("trace"),
            evaluators=params.get("evaluators"),
            type=LogType.GENERATION
        )

        super().__init__(base_log_params)

        self._prompt = params.get("prompt")
        self._input_tokens = params.get("input_tokens")
        self._output_tokens = params.get("output_tokens")
        self._cost = params.get("cost")

        # Convert variables to array format if needed
        variables = params.get("variables")
        if variables is not None:
            if isinstance(variables, dict):
                self._variables = [{"label": str(k), "value": str(v)} for k, v in variables.items()]
            elif isinstance(variables, list):
                self._variables = [{"label": str(v.get("label")), "value": str(v.get("value"))} for v in variables if v.get("label")]
            else:
                self._variables = []
        else:
            self._variables = []

        self._options = params.get("options")

    @property
    def prompt(self) -> Optional[Dict[str, Any]]:
        """Get the generation prompt."""
        return self._prompt

    @property
    def input_tokens(self) -> Optional[int]:
        """Get the generation input tokens."""
        return self._input_tokens

    @property
    def output_tokens(self) -> Optional[int]:
        """Get the generation output tokens."""
        return self._output_tokens

    @property
    def cost(self) -> Optional[float]:
        """Get the generation cost."""
        return self._cost

    @property
    def variables(self) -> List[Dict[str, str]]:
        """Get the generation variables."""
        return self._variables

    @property
    def options(self) -> Optional[Dict[str, Any]]:
        """Get the generation options."""
        return self._options

    @options.setter
    def options(self, options: Dict[str, Any]):
        """Set the generation options."""
        self._options = options

    def start(self, input: Optional['Input'] = None) -> 'Generation':
        """
        Start the generation with an optional input.

        Args:
            input (Optional[str]): The input to the generation.

        Returns:
            Generation: The generation instance.
        """
        if input:
            self._input = input

        super().start()
        return self

    def end(self,
            output: Optional[Output] = None,
            input_tokens: Optional[int] = None,
            output_tokens: Optional[int] = None,
            cost: Optional[float] = None
            ) -> 'Generation':
        """
        End the generation with an optional output or update parameters.

        Args:
            output (Optional[Output]): The output of the generation as string, array or a dictionary.
            input_tokens (Optional[int]): Optional number of tokens used for the input.
            output_tokens (Optional[int]): Optional number of tokens used for the output.
            cost (Optional[float]): Cost of the generation.

        Returns:
            Generation: The generation instance.
        """
        super().end()

        self._output = output
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._cost = cost

        # If this is a single generation, end the trace as well
        if self._options and self._options.get("type") == "single":
            self.trace.end_sync(self._output)

        return self

    def update(self, params: Dict[str, Any]) -> 'Generation':
        """
        Update the generation.

        Args:
            params (Dict[str, Any]): Parameters to update.

        Returns:
            Generation: The generation instance.
        """
        self._input = params.get("input", self._input)
        self._output = params.get("output", self._output)
        self._prompt = params.get("prompt", self._prompt)
        self._input_tokens = params.get("input_tokens", self._input_tokens)
        self._output_tokens = params.get("output_tokens", self._output_tokens)
        self._cost = params.get("cost", self._cost)

        # Update variables if provided
        variables = params.get("variables")
        if variables is not None:
            if isinstance(variables, dict):
                self._variables = [{"label": str(k), "value": str(v)} for k, v in variables.items()]
            elif isinstance(variables, list):
                self._variables = [{"label": str(v.get("label")), "value": str(v.get("value"))} for v in variables if v.get("label")]
            else:
                self._variables = []

        super().update(params)
        return self
