from typing import Dict, Optional, Any

from ..utils.protocols import IApi, ILogger
from ..utils.dtos import TraceParams, GenerationParams, LogParams

from ..objects.trace import Trace
from ..objects.generation import Generation
from ..objects.log import Log
from ..utils.flusher import Flusher

class MonitorSDK:
    """
    SDK for monitoring and tracing in Basalt.
    """
    def __init__(
            self,
            api: IApi,
            logger: ILogger
        ):
        self._api = api
        self._logger = logger

    def create_trace(
        self,
        slug: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Trace:
        """
        Creates a new trace for monitoring.

        Args:
            slug (str): The unique identifier for the trace.
            params (Optional[Dict[str, Any]]): Optional parameters for the trace.

        Returns:
            Trace: A new Trace instance.
        """
        if params is None:
            params = {}
            
        trace_params = TraceParams(**params)
        return self._create_trace(slug, trace_params)

    def create_generation(
        self,
        params: Dict[str, Any]
    ) -> Generation:
        """
        Creates a new generation for monitoring.

        Args:
            params (Dict[str, Any]): Parameters for the generation.

        Returns:
            Generation: A new Generation instance.
        """
        generation_params = GenerationParams(**params)
        return self._create_generation(generation_params)

    def create_log(
        self,
        params: Dict[str, Any]
    ) -> Log:
        """
        Creates a new log for monitoring.

        Args:
            params (Dict[str, Any]): Parameters for the log.

        Returns:
            Log: A new Log instance.
        """
        log_params = LogParams(**params)
        return self._create_log(log_params)

    def _create_trace(
        self,
        slug: str,
        params: TraceParams
    ) -> Trace:
        """
        Internal implementation for creating a trace.

        Args:
            slug (str): The unique identifier for the trace.
            params (TraceParams): Parameters for the trace.

        Returns:
            Trace: A new Trace instance.
        """
        flusher = Flusher(self._api, self._logger)
        # Convert TraceParams to a dictionary before passing to Trace
        params_dict = {
            "input": params.input,
            "output": params.output,
            "name": params.name,
            "start_time": params.start_time,
            "end_time": params.end_time,
            "user": params.user,
            "organization": params.organization,
            "metadata": params.metadata
        }
        trace = Trace(slug, params_dict, flusher)
        return trace

    def _create_generation(
        self,
        params: GenerationParams
    ) -> Generation:
        """
        Internal implementation for creating a generation.

        Args:
            params (GenerationParams): Parameters for the generation.

        Returns:
            Generation: A new Generation instance.
        """
        # Convert GenerationParams to a dictionary before passing to Generation
        params_dict = {
            "name": params.name,
            "trace": params.trace,
            "prompt": params.prompt,
            "input": params.input,
            "output": params.output,
            "variables": params.variables,
            "parent": params.parent,
            "metadata": params.metadata,
            "start_time": params.start_time,
            "end_time": params.end_time,
            "options": params.options
        }
        return Generation(params_dict)

    def _create_log(
        self,
        params: LogParams
    ) -> Log:
        """
        Internal implementation for creating a log.

        Args:
            params (LogParams): Parameters for the log.

        Returns:
            Log: A new Log instance.
        """
        # Convert LogParams to a dictionary before passing to Log
        params_dict = {
            "name": params.name,
            "trace": params.trace,
            "input": params.input,
            "output": params.output,
            "parent": params.parent,
            "metadata": params.metadata,
            "start_time": params.start_time,
            "end_time": params.end_time
        }
        return Log(params_dict) 