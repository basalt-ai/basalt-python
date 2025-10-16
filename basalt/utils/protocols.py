from collections.abc import Hashable, Mapping
from typing import Any, Literal, Protocol, TypeVar

from ..resources.monitor.monitorsdk_types import IMonitorSDK
from .dtos import (
    CreateDatasetItemResult,
    DescribeResult,
    GetDatasetResult,
    GetPromptResult,
    ListDatasetsResult,
    ListResult,
)

Input = TypeVar('Input')
Output = TypeVar('Output')


class ICache(Protocol):
    def get(self, key: Hashable) -> Any | None: ...

    def put(self, key: Hashable, value: Any, duration: int) -> None: ...


class IEndpoint(Protocol[Input, Output]):
    def prepare_request(self, dto: Input | None = None) -> dict[str, Any]: ...

    def decode_response(self, response: Any) -> tuple[Exception | None, Output | None]: ...


class IApi(Protocol):
    async def invoke(self, endpoint: IEndpoint[Input, Output], dto: Input | None = None) -> tuple[
        Exception | None, Output | None]: ...

    def invoke_sync(self, endpoint: IEndpoint[Input, Output], dto: Input | None = None) -> tuple[
        Exception | None, Output | None]: ...


class INetworker(Protocol):
    async def fetch(self,
                    url: str,
                    method: str,
                    body: Any | None = None,
                    params: Mapping[str, str] | None = None,
                    headers: Mapping[str, str] | None = None
                    ) -> tuple[Exception | None, Output | None]: ...

    def fetch_sync(self,
                   url: str,
                   method: str,
                   body: Any | None = None,
                   params: Mapping[str, str] | None = None,
                   headers: Mapping[str, str] | None = None
                   ) -> tuple[Exception | None, Output | None]: ...


class IPromptSDK(Protocol):
    async def get(self, slug: str,  version: str | None = None, tag: str | None = None,
                  variables: dict[str, str] | None = None, cache_enabled: bool = True) -> GetPromptResult: ...

    def get_sync(self, slug: str,  version: str | None = None, tag: str | None = None,
                 variables: dict[str, str] | None = None, cache_enabled: bool = True) -> GetPromptResult: ...

    async def describe(self, slug: str,  version: str | None = None, tag: str | None = None) -> DescribeResult: ...

    def describe_sync(self, slug: str,  version: str | None = None, tag: str | None = None) -> DescribeResult: ...

    async def list(self, feature_slug: str | None = None) -> ListResult: ...

    def list_sync(self, feature_slug: str | None = None) -> ListResult: ...


class IDatasetSDK(Protocol):
    async def list(self) -> ListDatasetsResult: ...

    def list_sync(self) -> ListDatasetsResult: ...

    async def get(self, slug: str) -> GetDatasetResult: ...

    def get_sync(self, slug: str) -> GetDatasetResult: ...

    async def add_row(self, slug: str, values: dict[str, str], name: str | None = None,
                      ideal_output: str | None = None,
                      metadata: dict[str, Any] | None = None) -> CreateDatasetItemResult: ...

    def add_row_sync(self, slug: str, values: dict[str, str], name: str | None = None,
                     ideal_output: str | None = None,
                     metadata: dict[str, Any] | None = None) -> CreateDatasetItemResult: ...


class IBasaltSDK(Protocol):
    @property
    def prompt(self) -> IPromptSDK: ...

    @property
    def monitor(self) -> IMonitorSDK: ...

    @property
    def datasets(self) -> IDatasetSDK: ...


class ILogger:
    def warn(self, message: str): ...

    def info(self, message: str): ...

    def error(self, message: str): ...


LogLevel = Literal["all", "warning", "none"]
