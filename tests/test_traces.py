from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from dynamo_routing_harness.constants import TRACE_SOURCES
from dynamo_routing_harness.traces import (
    TraceDownloadError,
    ensure_traces,
    prepare_benchmark_traces,
    trace_path,
    write_multiturn_trace,
)


def _sha256(data: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


class TraceTests(unittest.TestCase):
    def test_writes_deterministic_multiturn_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_multiturn_trace(
                Path(tmp),
                sessions=2,
                turns=3,
                input_tokens=256,
                output_tokens=256,
            )
            rows = [
                json.loads(line)
                for line in path.read_text().splitlines()
            ]
        self.assertEqual(len(rows), 6)
        self.assertEqual(
            [row["session_id"] for row in rows],
            ["session-000000"] * 3 + ["session-000001"] * 3,
        )
        self.assertTrue(all(row["input_length"] == 256 for row in rows))
        self.assertTrue(all(row["output_length"] == 256 for row in rows))

    def test_downloads_missing_traces_and_verifies(self) -> None:
        payloads = {
            workload: f"{workload}-trace".encode() for workload in TRACE_SOURCES
        }
        url_to_workload = {
            source["url"]: workload
            for workload, source in TRACE_SOURCES.items()
        }
        originals = {
            workload: source["sha256"]
            for workload, source in TRACE_SOURCES.items()
        }
        calls: list[str] = []

        def fake_downloader(url: str) -> bytes:
            calls.append(url)
            return payloads[url_to_workload[url]]

        try:
            for workload, data in payloads.items():
                TRACE_SOURCES[workload]["sha256"] = _sha256(data)
            with tempfile.TemporaryDirectory() as tmp:
                trace_dir = Path(tmp)
                resolved = ensure_traces(
                    trace_dir,
                    downloader=fake_downloader,
                )
                for workload in TRACE_SOURCES:
                    path = trace_path(trace_dir, workload)
                    self.assertEqual(resolved[workload], path)
                    self.assertEqual(path.read_bytes(), payloads[workload])
                self.assertEqual(len(calls), len(TRACE_SOURCES))
        finally:
            for workload, sha in originals.items():
                TRACE_SOURCES[workload]["sha256"] = sha

    def test_existing_valid_trace_is_not_redownloaded(self) -> None:
        originals = {
            workload: source["sha256"]
            for workload, source in TRACE_SOURCES.items()
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                trace_dir = Path(tmp)
                for workload in TRACE_SOURCES:
                    data = f"{workload}-bytes".encode()
                    TRACE_SOURCES[workload]["sha256"] = _sha256(data)
                    trace_path(trace_dir, workload).write_bytes(data)

                def fail_downloader(url: str) -> bytes:
                    raise AssertionError(f"unexpected download: {url}")

                resolved = ensure_traces(
                    trace_dir,
                    downloader=fail_downloader,
                )
                self.assertEqual(len(resolved), len(TRACE_SOURCES))
        finally:
            for workload, sha in originals.items():
                TRACE_SOURCES[workload]["sha256"] = sha

    def test_checksum_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_dir = Path(tmp)
            workload = next(iter(TRACE_SOURCES))
            with self.assertRaises(TraceDownloadError):
                ensure_traces(
                    trace_dir,
                    workloads=(workload,),
                    downloader=lambda url: b"corrupted",
                )

    def test_unconfigured_workload_requires_local_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_dir = Path(tmp)
            with self.assertRaises(TraceDownloadError):
                ensure_traces(
                    trace_dir,
                    workloads=("unconfigured",),
                    downloader=lambda url: b"",
                )

    def test_prepares_capped_trace_without_increasing_short_requests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            source_dir.mkdir()
            source = trace_path(source_dir, "conversation")
            source.write_text(
                "\n".join(
                    json.dumps({"output_length": value, "input_length": 1})
                    for value in (100, 1024, 2000)
                )
                + "\n"
            )
            resolved, evidence = prepare_benchmark_traces(
                source_dir,
                root / "effective",
                1024,
                workloads=("conversation",),
            )
            records = [
                json.loads(line)
                for line in resolved["conversation"].read_text().splitlines()
            ]
        self.assertEqual(
            [record["output_length"] for record in records],
            [100, 1024, 1024],
        )
        self.assertEqual(evidence["conversation"]["capped_records"], 1)
        self.assertEqual(
            evidence["conversation"]["effective_output_tokens"]["max"],
            1024,
        )
        self.assertEqual(evidence["conversation"]["sessions"], 3)
        self.assertEqual(
            evidence["conversation"]["turns_per_session"]["max"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
