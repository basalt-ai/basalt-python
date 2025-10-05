from datetime import datetime
from typing import Dict, Optional, Any, List
import uuid

from ..ressources.monitor.base_log_types import BaseLogParams, LogType
from ..ressources.monitor.evaluator_types import Evaluator
from ..ressources.monitor.trace_types import Trace
from ..ressources.monitor.log_types import Log


class BaseLog:
    """
    Base class for logs and generations.
    """
    def __init__(self, params: BaseLogParams):
        self._id = f"log-{uuid.uuid4().hex[:8]}"
        self._type = params.type
        self._name = params.name
        self._start_time = params.start_time if params.start_time is not None else datetime.now()
        self._end_time = params.end_time
        self._metadata = params.metadata
        self._trace = params.trace
        self._parent = params.parent
        self._evaluators = params.evaluators
        self._ideal_output = params.ideal_output

        # Add to trace's logs list if trace exists
        if self._trace:
            self._trace.logs.append(self)

    @property
    def id(self) -> str:
        """Get the log ID."""
        return self._id

    @property
    def parent(self) -> Optional['Log']:
        """Get the parent log."""
        return self._parent

    @parent.setter
    def parent(self, parent: 'Log'):
        """Set the parent log."""
        self._parent = parent

    @property
    def type(self) -> LogType:
        """Get the log type."""
        return self._type

    @property
    def name(self) -> str:
        """Get the log name."""
        return self._name

    @property
    def ideal_output(self) -> Optional[str]:
        """Get the ideal output."""
        return self._ideal_output

    @property
    def start_time(self) -> datetime:
        """Get the start time."""
        return self._start_time

    @property
    def end_time(self) -> Optional[datetime]:
        """Get the end time."""
        return self._end_time

    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        """Get the metadata."""
        return self._metadata

    @property
    def trace(self) -> 'Trace':
        """Get the trace."""
        return self._trace

    @property
    def evaluators(self) -> List[Evaluator]:
        """Get the evaluators."""
        return self._evaluators

    @trace.setter
    def trace(self, trace: 'Trace'):
        """Set the trace."""
        self._trace = trace

    def start(self) -> 'BaseLog':
        """Start the log."""
        self._start_time = datetime.now()
        return self

    def set_metadata(self, metadata: Dict[str, Any]) -> 'BaseLog':
        """Set the metadata."""
        self._metadata = metadata
        return self

    def add_evaluator(self, evaluator: Evaluator) -> 'BaseLog':
        if self._evaluators is None:
            self._evaluators = []

        self._evaluators.append(evaluator)
        return self

    def set_ideal_output(self, ideal_output: str) -> 'BaseLog':
        """Set the ideal output."""
        self._ideal_output = ideal_output
        return self

    def update(self, params: Dict[str, Any]) -> 'BaseLog':
        """Update the log."""
        self._name = params.get("name", self._name)
        self._metadata = params.get("metadata", self._metadata)

        if params.get("start_time"):
            self._start_time = params.get("start_time")

        if params.get("end_time"):
            self._end_time = params.get("end_time")

        return self

    def end(self) -> 'BaseLog':
        """End the log."""
        self._end_time = datetime.now()
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert the log to a dictionary for API serialization."""
        return {
            "id": self._id,
            "type": self._type,
            "name": self._name,
            "ideal_output": self._ideal_output,
            "start_time": self._start_time,
            "end_time": self._end_time,
            "metadata": self._metadata,
            "parent": {"id": self._parent.id} if self._parent else None,
        }
