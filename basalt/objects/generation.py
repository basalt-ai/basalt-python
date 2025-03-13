from typing import Dict, Optional, Any

from .base_log import BaseLog

class Generation(BaseLog):
    """
    Class representing a generation in the monitoring system.
    """
    def __init__(self, params: Dict[str, Any]):
        params_with_type = {
            "type": "generation",
            **params
        }
        super().__init__(params_with_type)
        
        self._prompt = params.get("prompt")
        self._input = params.get("input")
        self._output = params.get("output")
        self._variables = params.get("variables")
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
    def variables(self) -> Optional[Dict[str, Any]]:
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

    def end(self, output: Optional[str] = None) -> 'Generation':
        """
        End the generation with an optional output.
        
        Args:
            output (Optional[str]): The output of the generation.
            
        Returns:
            Generation: The generation instance.
        """
        super().end()
        
        if output:
            self._output = output
            
        # If this is a single generation, end the trace as well
        if self._options and self._options.get("type") == "single":
            self.trace.end(output)
            
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
        
        super().update(params)
        return self 