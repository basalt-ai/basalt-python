
from ..endpoints.monitor.create_experiment import CreateExperimentDTO, CreateExperimentEndpoint
from ..objects.experiment import Experiment
from ..objects.generation import Generation
from ..objects.log import Log
from ..objects.trace import Trace
from ..resources.monitor.experiment_types import ExperimentParams
from ..resources.monitor.generation_types import GenerationParams
from ..resources.monitor.log_types import LogParams
from ..resources.monitor.trace_types import TraceParams
from ..utils.dtos import CreateExperimentResult
from ..utils.flusher import Flusher
from ..utils.protocols import IApi, ILogger


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

    async def create_experiment(
            self,
            feature_slug: str,
            params: ExperimentParams
    ) -> CreateExperimentResult:
        """
        Asynchronously creates a new experiment for monitoring.

        Args:
            feature_slug (str): The feature slug for the experiment.
            params (Dict[str, Any]): Parameters for the experiment.

        Returns:
            Experiment: A new Experiment instance.
        """
        return await self._create_experiment(feature_slug, params)

    def create_experiment_sync(
            self,
            feature_slug: str,
            params: ExperimentParams
    ) -> CreateExperimentResult:
        """
        Synchronously creates a new experiment for monitoring.

        Args:
            feature_slug (str): The feature slug for the experiment.
            params (Dict[str, Any]): Parameters for the experiment.

        Returns:
            Experiment: A new Experiment instance.
        """
        return self._create_experiment_sync(feature_slug, params)

    def create_trace(
            self,
            slug: str,
            params: TraceParams | None = None
    ) -> Trace:
        """
        Creates a new trace for monitoring.

        Args:
            slug (str): The unique identifier for the trace.
            params (TraceParams): Parameters for the trace.

        Returns:
            Trace: A new Trace instance.
        """
        return self._create_trace(slug, params if params else {})

    def create_generation(
            self,
            params: GenerationParams
    ) -> Generation:
        """
        Creates a new generation for monitoring.

        Args:
            params (GenerationParams): Parameters for the generation.

        Returns:
            Generation: A new Generation instance.
        """
        return self._create_generation(params)

    def create_log(
            self,
            params: LogParams
    ) -> Log:
        """
        Creates a new log for monitoring.

        Args:
            params (LogParams): Parameters for the log.

        Returns:
            Log: A new Log instance.
        """
        return self._create_log(params)

    async def _create_experiment(
            self,
            feature_slug: str,
            params: ExperimentParams
    ) -> CreateExperimentResult:
        """
        Internal async implementation for creating an experiment.

        Args:
            feature_slug (str): The feature slug for the experiment.
            params (ExperimentParams): Parameters for the experiment.

        Returns:
            Experiment: A new Experiment instance.
        """
        dto = CreateExperimentDTO(
            feature_slug=feature_slug,
            name=params['name'],
        )

        # Call the API endpoint
        err, result = await self._api.invoke(CreateExperimentEndpoint, dto)

        if err is None:
            return None, Experiment(result.experiment)

        return err, None

    def _create_experiment_sync(
            self,
            feature_slug: str,
            params: ExperimentParams
    ) -> CreateExperimentResult:
        """
        Internal sync implementation for creating an experiment.

        Args:
            feature_slug (str): The feature slug for the experiment.
            params (ExperimentParams): Parameters for the experiment.

        Returns:
            Experiment: A new Experiment instance.
        """
        dto = CreateExperimentDTO(
            feature_slug=feature_slug,
            name=params['name'],
        )

        # Call the API endpoint
        err, result = self._api.invoke_sync(CreateExperimentEndpoint, dto)

        if err is None:
            return None, Experiment(result.experiment)

        return err, None

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

        trace = Trace(slug, params, flusher, self._logger)
        return trace

    @staticmethod
    def _create_generation(params: GenerationParams) -> Generation:
        """
        Internal implementation for creating a generation.

        Args:
            params (GenerationParams): Parameters for the generation.

        Returns:
            Generation: A new Generation instance.
        """
        return Generation(params)

    @staticmethod
    def _create_log(params: LogParams) -> Log:
        """
        Internal implementation for creating a log.

        Args:
            params (LogParams): Parameters for the log.

        Returns:
            Log: A new Log instance.
        """
        return Log(params)
