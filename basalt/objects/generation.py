from typing import Dict, Optional, Any, List, Union

from .base_log import BaseLog
from ..ressources.monitor.generation_types import GenerationParams

class Generation(BaseLog):
    """
    Class representing a generation in the monitoring system.
    """
    def __init__(self, params: GenerationParams):
        params_with_type = {
            "type": "generation",
            **params
        }
        super().__init__(params_with_type)
        
        self._prompt = params.get("prompt")
        self._input = params.get("input")
        self._output = params.get("output")
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
    def input(self) -> Optional[str]:
        """Get the generation input."""
        return self._input

    @property
    def output(self) -> Optional[str]:
        """Get the generation output."""
        return self._output

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

    def start(self, input: Optional[str] = None) -> 'Generation':
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

    def end(self, output: Optional[Union[str, Dict[str, Any]]] = None) -> 'Generation':
        """
        End the generation with an optional output or update parameters.
        
        Args:
            output (Optional[Union[str, Dict[str, Any]]]): The output of the generation
                or a dictionary of parameters to update.
            
        Returns:
            Generation: The generation instance.
        """
        super().end()
        
        if isinstance(output, dict):
            self.update(output)
        elif isinstance(output, str):
            self._output = output
        
        # If this is a single generation, end the trace as well
        if self._options and self._options.get("type") == "single":
            self.trace.end(self._output)
        
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