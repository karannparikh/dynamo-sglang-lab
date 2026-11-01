from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dynamo_routing_harness.aiperf import (
    AIPerfValidationError,
    _aiperf_environment,
    aiperf_command,
    trace_dataset_entries,
    validate_outputs,
)
from dynamo_routing_harness.constants import LOCAL_ROOT


class AIPerfTests(unittest.TestCase):
    def _artifacts(self, root: Path, count: int = 2) -> Path:
        payload = {
            "is_complete": True,
            "completed_request_count": {"avg": count},
            "request_count": {"avg": count},
            "request_error_rate": {"avg": 0},
            "error_summary": None,
            "osl_mismatch_count": {"avg": 0},
        }
        (root / "profile_export_aiperf.json").write_text(json.dumps(payload))
        rows = [
            {
                "metadata": {
                    "request_start_ns": index * 1_000_000_000,
                    "request_end_ns": index * 1_000_000_000 + 20_000_000,
                    "was_cancelled": False,
                    "context_overflow_skip": False,
                },
                "metrics": {
                    "time_to_first_token": {
                        "value": 10,
                        "unit": "ms",
                    }
                },
            }
            for index in range(count)
        ]
        (root / "profile_export.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        return root

    def test_strict_outputs_validate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = validate_outputs(self._artifacts(Path(directory)), 2)
        self.assertTrue(result["strict_completion"])
        self.assertEqual(result["successful"], 2)
        self.assertEqual(result["latency_ms_successful"]["p99"], 20)

    def test_request_count_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = self._artifacts(Path(directory), 1)
            with self.assertRaisesRegex(AIPerfValidationError, "mismatch"):
                validate_outputs(artifacts, 2)

    def test_request_errors_preserve_partial_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = self._artifacts(Path(directory), 2)
            summary_path = artifacts / "profile_export_aiperf.json"
            payload = json.loads(summary_path.read_text())
            payload["request_count"]["avg"] = 1
            payload["request_error_rate"]["avg"] = 50
            payload["error_summary"] = [{"count": 1, "type": "TimeoutError"}]
            summary_path.write_text(json.dumps(payload))
            rows_path = artifacts / "profile_export.jsonl"
            rows = [
                json.loads(line)
                for line in rows_path.read_text().splitlines()
            ]
            rows[1]["metrics"] = {}
            rows[1]["error"] = {"type": "TimeoutError"}
            rows_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            with self.assertRaises(AIPerfValidationError) as caught:
                validate_outputs(artifacts, 2)
        self.assertIsNotNone(caught.exception.summary)
        assert caught.exception.summary is not None
        self.assertEqual(caught.exception.summary["successful"], 1)
        self.assertEqual(caught.exception.summary["failed"], 1)
        self.assertEqual(caught.exception.summary["error_rate_pct"], 50)

    def test_command_uses_power_independent_trace_contract(self) -> None:
        command = aiperf_command(
            Path("/aiperf"),
            Path("/trace.jsonl"),
            Path("/artifacts"),
            20,
            100,
        )
        self.assertIn("--custom-dataset-type", command)
        self.assertIn("mooncake_trace", command)
        self.assertIn("--no-fixed-schedule", command)
        self.assertIn("--use-server-token-count", command)

    def test_multi_turn_trace_counts_sessions_as_dataset_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            trace.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "session_id": session,
                            "input_length": 256,
                            "output_length": 256,
                        }
                    )
                    for session in ("a", "a", "b", "b")
                )
                + "\n"
            )
            self.assertEqual(trace_dataset_entries(trace), 2)

    def test_multi_turn_command_stops_after_complete_sessions(self) -> None:
        command = aiperf_command(
            Path("/aiperf"),
            Path("/trace.jsonl"),
            Path("/artifacts"),
            20,
            16,
            2,
        )
        self.assertIn("--conversation-num", command)
        self.assertNotIn("--request-count", command)

    def test_offline_tokenizer_uses_populated_hub_cache(self) -> None:
        env = _aiperf_environment()
        self.assertEqual(env["HF_HUB_CACHE"], str(LOCAL_ROOT / "aiperf-hf"))
        self.assertEqual(
            env["TRANSFORMERS_CACHE"],
            str(LOCAL_ROOT / "aiperf-hf"),
        )
        self.assertNotEqual(env["HF_HOME"], env["HF_HUB_CACHE"])


if __name__ == "__main__":
    unittest.main()
