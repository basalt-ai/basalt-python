from datetime import datetime
from typing import Dict, Optional, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .base_log import BaseLog
    from .generation import Generation
    from ..utils.flusher import Flusher

class Trace:
    """
    Class representing a trace in the monitoring system.
    """
    def __init__(self, slug: str, params: Dict[str, Any], flusher: 'Flusher'):
        self._chain_slug = slug
        
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
        self._flushed_promise = None

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
    def chain_slug(self) -> str:
        """Get the chain slug."""
        return self._chain_slug

    @property
    def end_time(self) -> Optional[datetime]:
        """Get the end time."""
        return self._end_time

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
        self._end_time = datetime.now()
        
        # Send to the API using the flusher
        if not self._flushed_promise:
            self._flusher.flush_trace(self)
            
        return self 

    def to_dict(self) -> Dict[str, Any]:
        """Convert the trace to a dictionary for API serialization."""
        return {
            "chain_slug": self._chain_slug,
            "input": self._input,
            "output": self._output,
            "name": self._name,
            "start_time": self._start_time,
            "end_time": self._end_time,
            "user": self._user,
            "organization": self._organization,
            "metadata": self._metadata,
            "logs": self._logs
        }