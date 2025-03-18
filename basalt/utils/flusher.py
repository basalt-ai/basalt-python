from typing import TYPE_CHECKING, Dict, Any, List
import json
from datetime import datetime

if TYPE_CHECKING:
    from ..objects.trace import Trace
    from .protocols import IApi, ILogger

from ..endpoints.monitor.send_trace import SendTraceEndpoint

class Flusher:
    """
    Class for flushing traces to the API.
    """
    def __init__(self, api: 'IApi', logger: 'ILogger'):
        self._api = api
        self._logger = logger

    def _trace_to_dict(self, trace: 'Trace') -> Dict[str, Any]:
        """
        Convert a trace to a dictionary.
        
        Args:
            trace (Trace): The trace to convert.
            
        Returns:
            Dict[str, Any]: The trace as a dictionary.
        """
        output = trace.output
        if output is not None and not isinstance(output, str):
            output = json.dumps(output)
        return {
            "chain_slug": trace.chain_slug,
            "input": trace.input,
            "output": output,
            "name": trace._name,
            "start_time": trace.start_time.isoformat() if trace.start_time else None,
            "end_time": trace.end_time.isoformat() if trace.end_time else None,
            "user": trace.user,
            "organization": trace.organization,
            "metadata": trace.metadata,
            "logs": [self._log_to_dict(log) for log in trace.logs] if trace.logs else []
        }

    def _log_to_dict(self, log: Any) -> Dict[str, Any]:
        """
        Convert a log to a dictionary.
        
        Args:
            log (Any): The log to convert.
            
        Returns:
            Dict[str, Any]: The log as a dictionary.
        """
        # Convert output to string if it's an object
        output = log.output
        if output is not None and not isinstance(output, str):
            output = json.dumps(output)

        base_dict = {
            "id": log.id,
            "type": log.type,
            "name": log._name,
            "input": log.input,
            "output": output,
            "start_time": log.start_time.isoformat() if hasattr(log, 'start_time') and log.start_time else None,
            "end_time": log.end_time.isoformat() if hasattr(log, 'end_time') and log.end_time else None,
            "metadata": log.metadata if hasattr(log, 'metadata') else None,
            "parent": {"id": log.parent.id} if hasattr(log, 'parent') and log.parent else None
        }

        # Add generation-specific fields if it's a generation
        if log.type == "generation" and hasattr(log, "prompt"):
            base_dict.update({
                "prompt": log.prompt,
                "variables": log.variables if log.variables else [],  # Ensure variables is always a list
                "options": log.options if hasattr(log, "options") else None
            })

        return base_dict

    def flush_trace(self, trace: 'Trace') -> None:
        """
        Flush a trace to the API.
        
        Args:
            trace (Trace): The trace to flush.
        """
        try:
            if not self._api:
                self._logger.warn("Cannot flush trace: no API instance available")
                return
                
            # Create an endpoint instance
            endpoint = SendTraceEndpoint()
            
            # Convert trace to dictionary
            trace_dict = self._trace_to_dict(trace)
            
            # Create the DTO with the trace dictionary
            dto = {"trace": trace_dict}
            
            # Invoke the API with the endpoint and DTO
            error, result = self._api.invoke(endpoint, dto)
            
            if error:
                self._logger.warn(f"Failed to flush trace: {error}")
                return
                
            self._logger.warn(f"Successfully flushed trace {trace.chain_slug} to the API")
            
        except Exception as e:
            self._logger.warn(f"Exception while flushing trace: {str(e)}")