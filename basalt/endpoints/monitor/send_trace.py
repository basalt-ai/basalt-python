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
            logs = trace.get("logs", [])
        else:
            trace_data = trace.to_dict()
            # Convert logs to a format suitable for the API
            logs = []
            for log in trace_data["logs"]:
                dict_log = log.to_dict()

                # Convert dates and handle parent ID
                log_data = {
                    "startTime": dict_log["start_time"].isoformat() if isinstance(dict_log["start_time"], datetime) else dict_log["start_time"],
                    "endTime": dict_log["end_time"].isoformat() if isinstance(dict_log["end_time"], datetime) and dict_log["end_time"] else None,
                    "parentId": dict_log["parent"]["id"] if dict_log["parent"] else None,
                    "inputTokens": dict_log["input_tokens"] if "input_tokens" in dict_log else None,
                    "outputTokens": dict_log["output_tokens"] if "output_tokens" in dict_log else None,
                    "cost": dict_log["cost"] if "cost" in dict_log else None,
                    "variables": [{"label": k, "value": v} for k, v in dict_log["variables"].items()] if "variables" in dict_log else None,
                    "input": dict_log["input"] if "input" in dict_log else None,
                    "output": dict_log["output"] if "output" in dict_log else None,
                    "prompt": dict_log["prompt"] if "prompt" in dict_log else None,
                    "evaluators": dict_log["evaluators"] if "evaluators" in dict_log else None,
                    "idealOutput": dict_log["ideal_output"] if "ideal_output" in dict_log else None
                }

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
                    processed_log["endTime"] = processed_log["end_time"].isoformat() if isinstance(processed_log["end_time"], datetime) else processed_log["end_time"]
                    del processed_log["end_time"]

                # Extract parent ID
                if "parent" in processed_log and processed_log["parent"]:
                    processed_log["parentId"] = processed_log["parent"]["id"]
                    del processed_log["parent"]
                else:
                    processed_log["parentId"] = None

                # Rename ideal output
                if "ideal_output" in processed_log:
                  processed_log["idealOutput"] = processed_log["ideal_output"]
                  del processed_log["ideal_output"]

                # Rename input tokens
                if "input_tokens" in processed_log:
                    processed_log["inputTokens"] = processed_log["input_tokens"]
                    del processed_log["input_tokens"]

                # Rename output tokens
                if "output_tokens" in processed_log:
                    processed_log["outputTokens"] = processed_log["output_tokens"]
                    del processed_log["output_tokens"]

                processed_logs.append(processed_log)

        # Create the request body
        body = {
            "featureSlug": trace_data.get("feature_slug", trace_data.get("featureSlug")),
            "name": trace_data.get("name", trace_data.get("name")),
            "experiment": {"id": trace_data.get("experiment", {}).id} if trace_data.get("experiment") else None,
            "input": trace_data.get("input"),
            "output": trace_data.get("output"),
            "idealOutput": trace_data.get("ideal_output"),
            "metadata": trace_data.get("metadata"),
            "organization": trace_data.get("organization"),
            "user": trace_data.get("user"),
            "startTime": trace_data.get("start_time", trace_data.get("startTime")),
            "endTime": trace_data.get("end_time", trace_data.get("endTime")),
            "logs": processed_logs,
            "evaluators": trace_data.get("evaluators"),
            "evaluationConfig": trace_data.get("evaluationConfig")
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
