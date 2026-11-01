from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dynamo_routing_harness.constants import H200_PROFILE, expected_cases, topology
from dynamo_routing_harness.report import generate_report


class ReportTests(unittest.TestCase):
    def test_complete_matrix_generates_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            sources = {
                "dynamo": {
                    "path": "/dynamo",
                    "head": "d" * 40,
                    "dirty": False,
                },
                "fake_sglang": {
                    "path": "/fake-sglang",
                    "head": "f" * 40,
                    "dirty": True,
                },
            }
            matrix = {
                "offered_rps": 20,
                "h200_profile_from_application_defaults": H200_PROFILE,
                "topology": topology(),
                "runtime_image": "local/image:test",
                "planner_image": "local/planner:test",
                "sources": sources,
                "router_replica_sync": False,
                "output_token_limit": 1024,
                "trace_evidence": {
                    workload: {
                        "records": 3,
                        "sessions": 3,
                        "turns_per_session": {"min": 1, "max": 1, "mean": 1},
                        "input_tokens": {"min": 1, "max": 1},
                        "capped_records": 1,
                        "original_output_tokens": {"min": 1, "max": 2000},
                        "effective_output_tokens": {"min": 1, "max": 1024},
                        "effective_sha256": workload,
                    }
                    for workload in ("conversation", "toolagent")
                },
                "cases_finished": True,
                "comparison_complete": True,
                "expected_cases": expected_cases(),
            }
            (output / "matrix.json").write_text(json.dumps(matrix))
            for case in expected_cases():
                case_dir = output / case["case"]
                case_dir.mkdir()
                payload = {
                    **case,
                    "strict_completion": True,
                    "successful": 1,
                    "errors": {},
                    "dispatch_rps": 20,
                    "fleet_stable": True,
                    "prefill_cache": {"token_hit_fraction": 0.5},
                    "latency_ms_successful": {"p99": 10},
                    "ttft_ms_successful": {"p99": 2},
                }
                (case_dir / "summary.json").write_text(json.dumps(payload))
            summary = generate_report(output)
            markdown = (output / "report.md").read_text()
        self.assertTrue(summary["matrix_complete"])
        self.assertIn("`" + "d" * 40 + "`", markdown)
        self.assertIn("50.00%", markdown)
        self.assertIn("power-of-two", markdown)
        self.assertIn("Requested output-token cap: `1024`", markdown)
        self.assertIn("Router replica synchronization: `disabled`", markdown)
        self.assertIn(
            "| conversation | 3 | 3 | 1.00 (1-1) | 1-1 | 1 | 2000 | 1024 |",
            markdown,
        )
        self.assertIn("| conversation | KV-aware | off |", markdown)

    def test_failed_request_count_is_rendered(self) -> None:
        summary = {
            "sources": {
                "dynamo": {"head": "d", "dirty": False, "path": "/d"},
                "fake_sglang": {"head": "f", "dirty": False, "path": "/f"},
            },
            "offered_rps": 20,
            "cases": [
                {
                    "workload": "toolagent",
                    "routing_label": "load-aware",
                    "successful": 21766,
                    "failed": 1842,
                    "fleet_stable": False,
                }
            ],
        }
        from dynamo_routing_harness.report import build_markdown

        markdown = build_markdown(summary)
        self.assertIn(
            "| toolagent | load-aware | off | n/a | n/a | n/a | n/a | n/a | "
            "21766 | 1842 |",
            markdown,
        )


if __name__ == "__main__":
    unittest.main()
