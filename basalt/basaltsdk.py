from .utils.protocols import IPromptSDK, IBasaltSDK, IDatasetSDK
from .ressources.monitor.monitorsdk_types import IMonitorSDK

class BasaltSDK(IBasaltSDK):
    """
    The BasaltSDK class implements the IBasaltSDK interface.
    It serves as the main entry point for interacting with the Basalt SDK.
    
    """

    def __init__(self, prompt_sdk: IPromptSDK, monitor_sdk: IMonitorSDK, dataset_sdk: IDatasetSDK):
        self._prompt = prompt_sdk
        self._monitor = monitor_sdk
        self._datasets = dataset_sdk

    @property
    def prompt(self) -> IPromptSDK:
        """Read-only access to the PromptSDK instance"""
        return self._prompt

    @property
    def monitor(self) -> IMonitorSDK:
        """Read-only access to the MonitorSDK instance"""
        return self._monitor
        
    @property
    def datasets(self) -> IDatasetSDK:
        """Read-only access to the DatasetSDK instance"""
        return self._datasets
