from .basaltsdk import BasaltSDK
from .config import config
from .datasets.client import DatasetsClient
from .prompts.client import PromptsClient
from .resources.monitor.monitorsdk_types import IMonitorSDK
from .sdk.adapters.dataset_adapter import DatasetSDKAdapter
from .sdk.adapters.prompt_adapter import PromptSDKAdapter
from .sdk.monitorsdk import MonitorSDK
from .utils.api import Api
from .utils.logger import Logger
from .utils.memcache import MemoryCache
from .utils.networker import Networker
from .utils.protocols import IBasaltSDK, ICache, IDatasetSDK, IPromptSDK, LogLevel

global_fallback_cache = MemoryCache()


class BasaltFacade(IBasaltSDK):
    """
    The Basalt client facade providing unified access to SDK components.

    This class serves as the main entry point for interacting with Basalt services,
    providing access to prompts, monitoring, and datasets through a single interface.

    Example:
        >>> client = BasaltFacade(api_key="your-key")
        >>> prompt = client.prompt.get("my-prompt")
    """

    def __init__(
            self,
            api_key: str,
            log_level: LogLevel = "all",
            cache: ICache | None = None,
    ):
        """
        Initializes the Basalt client with the given API key and log level.

        Args:
            api_key (str): The API key for authenticating with the Basalt SDK.
            log_level (str, optional): The log level for the logger. Defaults to 'all'. (all, warn, error, debug, none)
            cache (ICache, optional): The cache to use for the SDK. Defaults to None, which means a MemoryCache will be used.

        Raises:
            ValueError: If the API key is empty.
        """

        if not api_key or not api_key.strip():
            raise ValueError("API key cannot be empty")

        if cache is None:
            cache = MemoryCache()

        logger = Logger(log_level=log_level)
        networker = Networker()

        api = Api(
            networker=networker,
            root_url=config["api_url"],
            api_key=api_key,
            sdk_version=config["sdk_version"],
            sdk_type=config["sdk_type"],
            logger=logger
        )

        # Create new clients
        prompts_client = PromptsClient(
            api_key=api_key,
            cache=cache,
            fallback_cache=global_fallback_cache,
            logger=logger,
            base_url=config["api_url"]
        )

        datasets_client = DatasetsClient(
            api_key=api_key,
            logger=logger,
            base_url=config["api_url"]
        )

        # Wrap new clients in adapters for backward compatibility
        prompt = PromptSDKAdapter(prompts_client, api, logger)
        datasets = DatasetSDKAdapter(datasets_client)
        monitor = MonitorSDK(api, logger)

        self._basalt = BasaltSDK(prompt, monitor, datasets)

    @property
    def prompt(self) -> IPromptSDK:
        """
        Read-only access to the PromptSDK instance.
        """
        return self._basalt.prompt

    @property
    def monitor(self) -> IMonitorSDK:
        """
        Read-only access to the MonitorSDK instance.
        """
        return self._basalt.monitor

    @property
    def datasets(self) -> IDatasetSDK:
        """
        Read-only access to the DatasetSDK instance.
        """
        return self._basalt.datasets
