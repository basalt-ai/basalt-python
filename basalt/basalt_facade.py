from .utils.api import Api
from .utils.protocols import IPromptSDK, IBasaltSDK, IMonitorSDK
from .sdk.promptsdk import PromptSDK
from .sdk.monitorsdk import MonitorSDK
from .basaltsdk import BasaltSDK
from .utils.memcache import MemoryCache
from .utils.networker import Networker
from .config import config
from .utils.logger import Logger

global_fallback_cache = MemoryCache()

class BasaltFacade(IBasaltSDK):
    """
    The Basalt client.
    """

    def __init__(self, api_key: str, log_level: str = 'all'):
        """
        Initializes the Basalt client with the given API key and log level.

        Args:
            api_key (str): The API key for authenticating with the Basalt SDK.
            log_level (str, optional): The log level for the logger. Defaults to 'all'. (all, warn, error, debug, none)
        """
        cache = MemoryCache()
        logger = Logger(log_level=log_level)
        networker = Networker(logger=logger)
        
        api = Api(
            networker=networker,
            root_url=config["api_url"],
            api_key=api_key,
            sdk_version=config["sdk_version"],
            sdk_type=config["sdk_type"],
            logger=logger
        )

        prompt = PromptSDK(api, cache, global_fallback_cache, logger)
        monitor = MonitorSDK(api, logger)

        self._basalt = BasaltSDK(prompt, monitor)

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
