from .protocols import IPromptSDK, IBasaltSDK

class BasaltSDK(IBasaltSDK):
    """
    The BasaltSDK class implements the IBasaltSDK interface.
    It serves as the main entry point for interacting with the Basalt SDK.
    """

    def __init__(self, prompt_sdk: IPromptSDK):
        self._prompt = prompt_sdk
    
    @property
    def prompt(self) -> IPromptSDK:
        """Read-only access to the PromptSDK instance"""
        return self._prompt

