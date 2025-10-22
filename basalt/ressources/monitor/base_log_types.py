from datetime import datetime
from typing import Dict, List, Optional, Union, Any, TYPE_CHECKING, TypedDict, Literal, TypeVar
from dataclasses import dataclass, field
from uuid import uuid4
from enum import Enum

if TYPE_CHECKING:
    from .log_types import Log

from .evaluator_types import Evaluator
from .trace_types import Trace

SelfType = TypeVar('SelfType', bound='BaseLog')

class LogType(str, Enum):
    """Enum-like class for log types.

    Attributes:
        SPAN: Represents a span log type
        GENERATION: Represents a generation log type
        FUNCTION: Represents a function log type
        TOOL: Represents a tool log type
        RETRIEVAL: Represents a retrieval log type
        EVENT: Represents an event log type
    """
    SPAN = 'span'
    GENERATION = 'generation'
    FUNCTION = 'function'
    TOOL = 'tool'
    RETRIEVAL = 'retrieval'
    EVENT = 'event'

# Type alias for log type strings - use this in TypedDict parameters
LogTypeStr = Literal['span', 'generation', 'function', 'tool', 'retrieval', 'event']

Input = str | List[Any] | Dict[str, Any]
Output = str | List[Any] | Dict[str, Any]
IdealOutput = str | List[Any] | Dict[str, Any]

class _BaseLogParamsRequired(TypedDict):
    """Required fields for BaseLogParams."""
    name: str

class BaseLogParams(_BaseLogParamsRequired, TypedDict, total=False):
    """Base parameters for creating a log entry.

    Attributes:
        name: Name of the log entry, describing what it represents (required).
        start_time: When the log entry started, can be a datetime object or ISO string.
            If not provided, defaults to the current time when created.
        end_time: When the log entry ended, can be a datetime object or ISO string.
            Can be set later using the end() method.
        metadata: Additional contextual information about this log entry.
            Can be any structured data relevant to the operation being logged.
        parent: Optional parent span if this log is part of a larger operation.
            Used to establish hierarchical relationships between operations.
        trace: The trace this log belongs to, providing the overall context.
            Every log must be associated with a trace.
        evaluators: The evaluators to attach to the log.
    """
    input: Optional[Input]
    output: Optional[Output]
    ideal_output: Optional[IdealOutput]
    start_time: Optional[Union[datetime, str]]
    end_time: Optional[Union[datetime, str]]
    metadata: Optional[Dict[str, Any]]
    parent: Optional['Log']
    trace: Optional['Trace']
    evaluators: List[Evaluator]

class BaseLogParamsWithType(BaseLogParams, TypedDict, total=False):
    """Base parameters for creating a log entry."""
    type: LogType

@dataclass
class BaseLog:
    """Base class for all log entries.

    Attributes:
        id: Unique identifier for this log entry.
            Automatically generated when the log is created.
        type: The type of log entry (e.g., 'span', 'generation').
            Used to distinguish between different kinds of logs.
        name: Name of the log entry, describing what it represents.
        start_time: When the log entry started, can be a datetime object or ISO string.
            If not provided, defaults to the current time when created.
        end_time: When the log entry ended, can be a datetime object or ISO string.
            Can be set later using the end() method.
        metadata: Additional contextual information about this log entry.
            Can be any structured data relevant to the operation being logged.
        parent: Optional parent span if this log is part of a larger operation.
            Used to establish hierarchical relationships between operations.
        trace: The trace this log belongs to, providing the overall context.
            Every log must be associated with a trace.
        evaluators: List of evaluators attached to the log.
    """
    input: Optional[Input]
    output: Optional[Output]
    ideal_output: Optional[IdealOutput]
    name: str
    type: LogType
    id: str = field(default_factory=lambda: str(f'log-{uuid4().hex[:8]}'))
    start_time: Optional[Union[datetime, str]] = None
    end_time: Optional[Union[datetime, str]] = None
    metadata: Optional[Dict[str, Any]] = None
    parent: Optional['Log'] = None
    trace: Optional['Trace'] = None
    evaluators: List[Evaluator] = field(default_factory=list)

    def start(self: SelfType, input: Optional[Input] = None) -> SelfType:
        """Marks the log as started and sets the start time if not already set.

        Returns:
            The log instance for method chaining.
        """
        ...

    def set_metadata(self: SelfType, metadata: Optional[Dict[str, Any]] = None) -> SelfType:
        """Sets the metadata for the log.

        Args:
            metadata: The metadata to set for the log.

        Returns:
            The log instance for method chaining.
        """
        ...

    def add_evaluator(self: SelfType, evaluator: Evaluator) -> SelfType:
        """Adds an evaluator to the log.

        Args:
            evaluator: The evaluator to add to the log.

        Returns:
            The log instance for method chaining.
        """
        ...

    def update(self: SelfType, params: Dict[str, Any]) -> SelfType:
        """Updates the log with new parameters.

        Args:
            **params: The parameters to update.

        Returns:
            The log instance for method chaining.
        """
        ...

    def end(self: SelfType, **kwargs) -> SelfType:
        """Marks the log as ended.

        Returns:
            The log instance for method chaining.
        """
        ...
