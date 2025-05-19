class LogType:
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