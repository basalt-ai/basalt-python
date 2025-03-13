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
        # Remove child log from the list of its previous trace
        generation.trace.logs = [log for log in generation.trace.logs if log.id != generation.id]
        
        # Add child to the new trace list
        self.trace.logs.append(generation)
        
        # Set the trace of the generation to the current log's trace
        generation.trace = self.trace
        generation.options = {"type": "multi"}
        
        # Set the parent of the generation to the current log
        generation.parent = self
        
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
        
        # Set the name to the prompt slug if available
        name = params.get("name")
        if params.get("prompt") and params["prompt"].get("slug"):
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