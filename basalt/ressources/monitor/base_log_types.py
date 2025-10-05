from datetime import datetime
from typing import Dict, List, Optional, Union, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from uuid import uuid4
from enum import Enum

if TYPE_CHECKING:
    from .log_types import Log

from .evaluator_types import Evaluator
from .trace_types import Trace


class LogType(Enum):
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

@dataclass
class BaseLogParams:
    """Base parameters for creating a log entry.

    Attributes:
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
        evaluators: The evaluators to attach to the log.
    """
    name: str
    ideal_output: Optional[str] = None
    start_time: Optional[Union[datetime, str]] = None
    end_time: Optional[Union[datetime, str]] = None
    metadata: Optional[Dict[str, Any]] = None
    parent: Optional['Log'] = None
    trace: 'Trace' = None
    evaluators: Optional[List[Evaluator]] = None
    type: Optional['LogType'] = None

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
    id: str = field(default_factory=lambda: str(f'log-{uuid4().hex[:8]}'))
    type: LogType = None
    name: str = None
    start_time: Optional[Union[datetime, str]] = None
    end_time: Optional[Union[datetime, str]] = None
    metadata: Optional[Dict[str, Any]] = None
    parent: Optional['Log'] = None
    trace: 'Trace' = None
    evaluators: List[Evaluator] = field(default_factory=list)

    def start(self) -> 'BaseLog':
        """Marks the log as started and sets the start time if not already set.

        Returns:
            The log instance for method chaining.
        """
        ...

    def set_metadata(self, metadata: Optional[Dict[str, Any]] = None) -> 'BaseLog':
        """Sets the metadata for the log.

        Args:
            metadata: The metadata to set for the log.

        Returns:
            The log instance for method chaining.
        """
        ...

    def add_evaluator(self, evaluator: Evaluator) -> 'BaseLog':
        """Adds an evaluator to the log.

        Args:
            evaluator: The evaluator to add to the log.

        Returns:
            The log instance for method chaining.
        """
        ...

    def update(self, params: Dict[str, Any]) -> 'BaseLog':
        """Updates the log with new parameters.

        Args:
            **params: The parameters to update.

        Returns:
            The log instance for method chaining.
        """
        ...

    def end(self) -> 'BaseLog':
        """Marks the log as ended.

        Returns:
            The log instance for method chaining.
        """
        ...
