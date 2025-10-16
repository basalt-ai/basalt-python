from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Optional, TypedDict, TypeVar
from uuid import uuid4

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
    ideal_output: str | None
    start_time: datetime | str | None
    end_time: datetime | str | None
    metadata: dict[str, Any] | None
    parent: Optional['Log']
    trace: Optional['Trace']
    evaluators: list[Evaluator] | None


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
    name: str
    type: LogType
    id: str = field(default_factory=lambda: str(f'log-{uuid4().hex[:8]}'))
    start_time: datetime | str | None = None
    end_time: datetime | str | None = None
    metadata: dict[str, Any] | None = None
    parent: Optional['Log'] = None
    trace: Optional['Trace'] = None
    evaluators: list[Evaluator] = field(default_factory=list)

    def start(self: SelfType) -> SelfType:
        """Marks the log as started and sets the start time if not already set.

        Returns:
            The log instance for method chaining.
        """
        ...

    def set_metadata(self: SelfType, metadata: dict[str, Any] | None = None) -> SelfType:
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

    def update(self: SelfType, params: dict[str, Any]) -> SelfType:
        """Updates the log with new parameters.

        Args:
            **params: The parameters to update.

        Returns:
            The log instance for method chaining.
        """
        ...

    def end(self: SelfType) -> SelfType:
        """Marks the log as ended.

        Returns:
            The log instance for method chaining.
        """
        ...
