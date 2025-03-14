from typing import Dict, Optional, Any, TYPE_CHECKING

from .base_log import BaseLog

if TYPE_CHECKING:
    from .generation import Generation

class Log(BaseLog):
    """
    Class representing a log in the monitoring system.
    """
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self._input = params.get("input")
        self._output = None

    @property
    def input(self) -> Optional[str]:
        """Get the log input."""
        return self._input

    @property
    def output(self) -> Optional[str]:
        """Get the log output."""
        return self._output

    def start(self, input: Optional[str] = None) -> 'Log':
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

    def end(self, output: Optional[str] = None) -> 'Log':
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
        generation.parent = self
        generation.trace = self.trace
        
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

    def create_generation(self, params: Dict[str, Any]) -> 'Generation':
        """
        Create a new generation as a child of this log.
        
        Args:
            params (Dict[str, Any]): Parameters for the generation.
            
        Returns:
            Generation: The new generation instance.
        """
        from .generation import Generation
        
        # Set the name to the prompt slug only if name is not provided
        name = params.get("name")
        if not name and params.get("prompt") and params["prompt"].get("slug"):
            name = params["prompt"]["slug"]
            
        generation = Generation({
            **params,
            "name": name,
            "trace": self.trace,
            "parent": self
        })
        
        return generation

    def create_log(self, params: Dict[str, Any]) -> 'Log':
        """
        Create a new log as a child of this log.
        
        Args:
            params (Dict[str, Any]): Parameters for the log.
            
        Returns:
            Log: The new log instance.
        """
        log = Log({
            **params,
            "trace": self.trace,
            "parent": self
        })
        
        return log 