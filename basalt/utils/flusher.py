from typing import TYPE_CHECKING
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
            
            # Create the DTO with the trace
            dto = {"trace": trace}
            
            # Invoke the API with the endpoint and DTO
            error, result = self._api.invoke(endpoint, dto)
            
            if error:
                self._logger.warn(f"Failed to flush trace: {error}")
                return
                
            self._logger.warn(f"Successfully flushed trace {trace.chain_slug} to the API")
            
        except Exception as e:
            self._logger.warn(f"Exception while flushing trace: {str(e)}") 