from datetime import datetime
from typing import Dict, Optional, Any, TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    from .log import Log
    from .trace import Trace

class BaseLog:
    """
    Base class for logs and generations.
    """
    def __init__(self, params: Dict[str, Any]):
        self._id = f"log-{uuid.uuid4().hex[:8]}"
        self._type = params.get("type")
        self._name = params.get("name")
        self._start_time = params.get("start_time", datetime.now())
        self._end_time = params.get("end_time")
        self._metadata = params.get("metadata")
        self._trace = params.get("trace")
        self._parent = params.get("parent")

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
    def type(self) -> str:
        """Get the log type."""
        return self._type

    @property
    def name(self) -> str:
        """Get the log name."""
        return self._name

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
            "start_time": self._start_time,
            "end_time": self._end_time,
            "metadata": self._metadata,
            "parent": {"id": self._parent.id} if self._parent else None,
        }