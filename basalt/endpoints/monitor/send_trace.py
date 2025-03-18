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
        
        # Check if trace is already a dictionary or an object
        if isinstance(trace, dict):
            trace_data = trace
            logs = trace_data.get("logs", [])
        else:
            trace_data = trace.to_dict()
            # Convert logs to a format suitable for the API
            logs = []
            for log in trace_data["logs"]:
                log_data = log.to_dict()
                
                # Convert dates to ISO format
                log_data["startTime"] = log_data["start_time"].isoformat() if isinstance(log_data["start_time"], datetime) else log_data["start_time"]
                log_data["endTime"] = log_data["end_time"].isoformat() if isinstance(log_data["end_time"], datetime) and log_data["end_time"] else None
                
                # Remove old format keys
                del log_data["start_time"]
                del log_data["end_time"]

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
                    
                # Extract parent ID
                if log_data["parent"]:
                    log_data["parentId"] = log_data["parent"]["id"]
                    del log_data["parent"]
                else:
                    log_data["parentId"] = None
                    
                logs.append(log_data)
        
        # Process logs if they're already in dictionary format
        processed_logs = []
        for log_data in logs:
            # If log_data is already processed by the flusher, it will have these keys
            if "startTime" in log_data and "endTime" in log_data:
                # Extract parent ID if it's in the parent format
                if "parent" in log_data and log_data["parent"]:
                    log_data["parentId"] = log_data["parent"]["id"]
                    del log_data["parent"]
                processed_logs.append(log_data)
            else:
                # Convert dates to ISO format if they're in the old format
                processed_log = dict(log_data)
                if "start_time" in processed_log:
                    processed_log["startTime"] = processed_log["start_time"].isoformat() if isinstance(processed_log["start_time"], datetime) else processed_log["start_time"]
                    del processed_log["start_time"]
                if "end_time" in processed_log:
                    processed_log["endTime"] = processed_log["end_time"].isoformat() if isinstance(processed_log["end_time"], datetime) and processed_log["end_time"] else None
                    del processed_log["end_time"]
                
                # Extract parent ID
                if "parent" in processed_log and processed_log["parent"]:
                    processed_log["parentId"] = processed_log["parent"]["id"]
                    del processed_log["parent"]
                else:
                    processed_log["parentId"] = None
                
                processed_logs.append(processed_log)
            
        # Create the request body
        body = {
            "chainSlug": trace_data.get("chain_slug", trace_data.get("chainSlug")),
            "input": trace_data.get("input"),
            "output": trace_data.get("output"),
            "metadata": trace_data.get("metadata"),
            "organization": trace_data.get("organization"),
            "user": trace_data.get("user"),
            "startTime": trace_data.get("start_time", trace_data.get("startTime")),
            "endTime": trace_data.get("end_time", trace_data.get("endTime")),
            "logs": processed_logs
        }
        
        # Convert dates to ISO format if they're datetime objects
        if isinstance(body["startTime"], datetime):
            body["startTime"] = body["startTime"].isoformat()
        if isinstance(body["endTime"], datetime):
            body["endTime"] = body["endTime"].isoformat()
        
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