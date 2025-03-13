"""
Endpoint for sending a trace to the API.
"""
from typing import Dict, Any, Optional, TypeVar, Tuple
from datetime import datetime

# Define type variables for the endpoint
Input = TypeVar('Input', bound=Dict[str, Any])
Output = TypeVar('Output', bound=Dict[str, Any])

class SendTraceEndpoint:
    """
    Endpoint for sending a trace to the API.
    """
    def prepare_request(self, dto: Optional[Input] = None) -> Dict[str, Any]:
        """
        Prepares the request for sending a trace.
        
        Args:
            dto (Optional[Dict[str, Any]]): The data transfer object containing the trace.
            
        Returns:
            Dict[str, Any]: The request information.
        """
        if not dto or "trace" not in dto:
            return {
                "method": "post",
                "path": "/monitor/trace",
                "body": {}
            }
            
        trace = dto["trace"]
        
        # Convert logs to a format suitable for the API
        logs = []
        for log in trace.logs:
            log_data = {
                "id": log.id,
                "type": log.type,
                "name": log.name,
                "startTime": log.start_time.isoformat() if isinstance(log.start_time, datetime) else log.start_time,
                "endTime": log.end_time.isoformat() if isinstance(log.end_time, datetime) and log.end_time else None,
                "metadata": log.metadata,
                "parentId": log.parent.id if log.parent else None,
            }
            
            # Add input and output if they exist
            if hasattr(log, "input"):
                log_data["input"] = log.input
            if hasattr(log, "output"):
                log_data["output"] = log.output
                
            # Add prompt and variables if it's a generation
            if hasattr(log, "prompt"):
                log_data["prompt"] = log.prompt
            if hasattr(log, "variables") and log.variables:
                log_data["variables"] = [{"label": key, "value": value} for key, value in log.variables.items()]
                
            logs.append(log_data)
            
        # Create the request body
        body = {
            "chainSlug": trace.chain_slug,
            "input": trace.input,
            "output": trace.output,
            "metadata": trace.metadata,
            "organization": trace.organization,
            "user": trace.user,
            "startTime": trace.start_time.isoformat() if isinstance(trace.start_time, datetime) else trace.start_time,
            "endTime": trace.end_time.isoformat() if isinstance(trace.end_time, datetime) and trace.end_time else None,
            "logs": logs
        }
        
        return {
            "method": "post",
            "path": "/monitor/trace",
            "body": body
        }
    
    def decode_response(self, response: Any) -> Tuple[Optional[Exception], Optional[Output]]:
        """
        Decodes the response from sending a trace.
        
        Args:
            response (Any): The response from the API.
            
        Returns:
            Tuple[Optional[Exception], Optional[Dict[str, Any]]]: The decoded response.
        """
        if not isinstance(response, dict):
            return Exception("Failed to decode response (invalid body format)"), None
            
        return None, response.get("trace", {}) 