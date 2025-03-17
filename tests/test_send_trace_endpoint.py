import unittest
from datetime import datetime
from basalt.endpoints.monitor.send_trace import SendTraceEndpoint


class TestSendTraceEndpoint(unittest.TestCase):
    def test_prepare_request_with_empty_dto(self):
        result = SendTraceEndpoint().prepare_request(None)

        self.assertEqual(result["method"], "post")
        self.assertEqual(result["path"], "/monitor/trace")
        self.assertEqual(result["body"], {})

    def test_prepare_request_with_full_trace(self):
        # Create a mock trace with all required fields
        trace = {
            "chain_slug": "test-chain",
            "input": {"query": "test"},
            "output": {"response": "test-response"},
            "metadata": {"test": "metadata"},
            "organization": "test-org",
            "user": "test-user",
            "start_time": datetime(2025, 3, 17, 12, 0),
            "end_time": datetime(2025, 3, 17, 12, 1),
            "logs": [
                {
                    "id": "log1",
                    "type": "generation",
                    "name": "test-log",
                    "start_time": datetime(2025, 3, 17, 12, 0),
                    "end_time": datetime(2025, 3, 17, 12, 1),
                    "metadata": {"log": "metadata"},
                    "parent": None,
                    "input": {"log": "input"},
                    "output": {"log": "output"},
                    "prompt": "test prompt",
                    "variables": [{"label": "var1", "value": "value1"}]
                }
            ]
        }

        result = SendTraceEndpoint().prepare_request({"trace": trace})

        # Verify the basic request structure
        self.assertEqual(result["method"], "post")
        self.assertEqual(result["path"], "/monitor/trace")
        
        # Verify the body contains all required fields
        body = result["body"]
        self.assertEqual(body["chainSlug"], "test-chain")
        self.assertEqual(body["input"], {"query": "test"})
        self.assertEqual(body["output"], {"response": "test-response"})
        self.assertEqual(body["metadata"], {"test": "metadata"})
        self.assertEqual(body["organization"], "test-org")
        self.assertEqual(body["user"], "test-user")
        self.assertEqual(body["startTime"], "2025-03-17T12:00:00")
        self.assertEqual(body["endTime"], "2025-03-17T12:01:00")

        # Verify logs are properly formatted
        self.assertEqual(len(body["logs"]), 1)
        log = body["logs"][0]
        self.assertEqual(log["id"], "log1")
        self.assertEqual(log["type"], "generation")
        self.assertEqual(log["name"], "test-log")
        self.assertEqual(log["startTime"], "2025-03-17T12:00:00")
        self.assertEqual(log["endTime"], "2025-03-17T12:01:00")
        self.assertEqual(log["metadata"], {"log": "metadata"})
        self.assertIsNone(log["parentId"])
        self.assertEqual(log["input"], {"log": "input"})
        self.assertEqual(log["output"], {"log": "output"})
        self.assertEqual(log["prompt"], "test prompt")
        self.assertEqual(log["variables"], [{"label": "var1", "value": "value1"}])

    def test_decode_valid_response(self):
        response = {
            "trace": {
                "id": "trace-123",
                "status": "success"
            }
        }

        exception, decoded = SendTraceEndpoint().decode_response(response)

        self.assertIsNone(exception)
        self.assertEqual(decoded, {"id": "trace-123", "status": "success"})

    def test_decode_invalid_response(self):
        exception, decoded = SendTraceEndpoint().decode_response("invalid")

        self.assertIsNotNone(exception)
        self.assertIsNone(decoded)
        self.assertEqual(str(exception), "Failed to decode response (invalid body format)")
