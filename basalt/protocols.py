from typing import Any, Optional, Protocol, Hashable, Tuple, TypeVar, Dict, Mapping
from .dtos import PromptResponse

Input = TypeVar('Input')
Output = TypeVar('Output')

class ICache(Protocol):
    def get(self, key: Hashable) -> Optional[Any]: ...
    def put(self, key: Hashable, value: Any, duration: int) -> None: ...

class IEndpoint(Protocol[Input, Output]):
    def prepare_request(self, dto: Input) -> Dict[str, Any]: ...
    def decode_response(self, response: Any) -> Tuple[Optional[Exception], Optional[Output]]: ...

class IApi(Protocol):
    def invoke(self,endpoint: IEndpoint[Input, Output], dto: Input) -> Tuple[Optional[Exception], Optional[Output]]: ...

class INetworker(Protocol):
    def fetch(self,
              url: str,
              method: str,
              body: Optional[Any] = None,
              params: Optional[Mapping[str, str]] = None,
              headers: Optional[Mapping[str, str]] = None
            ) -> Tuple[Optional[Exception], Optional[Output]]: ...

class IPromptSDK(Protocol):
    def get(self, slug: str, tag: Optional[str] = None, version: Optional[str] = None) -> PromptResponse: ...

class IBasaltSDK(Protocol):
    @property
    def prompt(self) -> IPromptSDK: ...

class ILogger:
    def warn(self): ...
