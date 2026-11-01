from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .command import Runner
from .constants import (
    EXPECTED_REPLICAS,
    FLEET_POLL_SECONDS,
    LOCAL_ROOT,
    MODEL_NAME,
    REPOSITORY_ROOT,
    TOKENIZER,
    TOKENIZER_REVISION,
)
from .kubernetes import FleetSnapshot, Kubernetes, validate_fleet


class AIPerfValidationError(RuntimeError):
    def __init__(
        self,
        message: str,
        summary: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.summary = summary


def _venv_interpreter() -> str:
    """Interpreter used to create the AIPerf venv.

    AIPerf requires Python >= 3.10, but the system ``python3`` may be older
    (macOS ships 3.9). Prefer an explicit newer interpreter on PATH, then an
    ``uv``-managed one, and only fall back to ``python3``.
    """
    for candidate in ("python3.13", "python3.12", "python3.11", "python3.10"):
        found = shutil.which(candidate)
        if found:
            return found
    uv = shutil.which("uv")
    if uv:
        result = subprocess.run(
            (uv, "python", "find", ">=3.10"),
            capture_output=True,
            text=True,
        )
        path = result.stdout.strip()
        if result.returncode == 0 and path:
            return path
    return "python3"


def setup_aiperf(runner: Runner | None = None) -> Path:
    active_runner = runner or Runner()
    venv = LOCAL_ROOT / "aiperf-venv"
    python = venv / "bin" / "python"
    if not python.exists():
        active_runner.run((_venv_interpreter(), "-m", "venv", venv))
    active_runner.run(
        (
            python,
            "-m",
            "pip",
            "install",
            "--requirement",
            REPOSITORY_ROOT / "benchmark" / "aiperf-requirements.txt",
        )
    )
    cache = LOCAL_ROOT / "aiperf-hf"
    cache.mkdir(parents=True, exist_ok=True)
    active_runner.run(
        (
            python,
            "-c",
            (
                "from transformers import AutoTokenizer;"
                f"AutoTokenizer.from_pretrained({TOKENIZER!r},"
                f" revision={TOKENIZER_REVISION!r}, cache_dir={os.fspath(cache)!r})"
            ),
        )
    )
    return venv / "bin" / "aiperf"


def trace_request_count(trace: Path) -> int:
    count = sum(1 for line in trace.read_text().splitlines() if line.strip())
    if count <= 0:
        raise AIPerfValidationError(f"trace has no requests: {trace}")
    return count


def trace_dataset_entries(trace: Path) -> int:
    identities: set[tuple[str, object]] = set()
    for line_number, line in enumerate(trace.read_text().splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        session_id = record.get("session_id")
        if isinstance(session_id, str) and session_id:
            identities.add(("session", session_id))
        else:
            identities.add(("row", line_number))
    if not identities:
        raise AIPerfValidationError(f"trace has no dataset entries: {trace}")
    return len(identities)


def _aiperf_environment() -> dict[str, str]:
    env = os.environ.copy()
    cache = LOCAL_ROOT / "aiperf-hf"
    env.update(
        {
            "HF_HOME": os.fspath(LOCAL_ROOT / "aiperf-hf-home"),
            "HF_HUB_CACHE": os.fspath(cache),
            "TRANSFORMERS_CACHE": os.fspath(cache),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONHASHSEED": "42",
            "AIPERF_DATASET_CONFIGURATION_TIMEOUT": "900",
            "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT": "960",
        }
    )
    return env


def aiperf_command(
    executable: Path,
    trace: Path,
    artifact_dir: Path,
    request_rate: float,
    request_count: int,
    dataset_entries: int | None = None,
) -> tuple[str, ...]:
    resolved_entries = dataset_entries or request_count
    stop_condition = (
        ("--conversation-num", str(resolved_entries))
        if resolved_entries < request_count
        else ("--request-count", str(request_count))
    )
    return (
        "/usr/bin/time",
        "-l",
        os.fspath(executable),
        "profile",
        "--model",
        MODEL_NAME,
        "--url",
        "http://127.0.0.1:8000",
        "--endpoint-type",
        "chat",
        "--streaming",
        "--tokenizer",
        TOKENIZER,
        "--tokenizer-revision",
        TOKENIZER_REVISION,
        "--input-file",
        os.fspath(trace),
        "--custom-dataset-type",
        "mooncake_trace",
        "--no-fixed-schedule",
        *stop_condition,
        "--num-dataset-entries",
        str(resolved_entries),
        "--dataset-sampling-strategy",
        "sequential",
        "--request-rate",
        str(request_rate),
        "--arrival-pattern",
        "constant",
        "--isl-block-size",
        "512",
        "--use-server-token-count",
        "--extra-inputs",
        '{"temperature":0,"ignore_eos":true}',
        "--request-timeout-seconds",
        "120",
        "--random-seed",
        "42",
        "--workers-max",
        "4",
        "--ui-type",
        "none",
        "--artifact-dir",
        os.fspath(artifact_dir),
    )


def _metric_avg(payload: dict[str, Any], name: str) -> float:
    value = payload.get(name)
    if not isinstance(value, dict) or "avg" not in value:
        raise AIPerfValidationError(f"AIPerf summary is missing {name}.avg")
    return float(value["avg"])


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p99": None, "max": None}
    ordered = sorted(values)

    def nearest(percentile: float) -> float:
        index = round((len(ordered) - 1) * percentile)
        return ordered[index]

    return {
        "p50": nearest(0.50),
        "p99": nearest(0.99),
        "max": max(ordered),
    }


def validate_outputs(
    artifact_dir: Path,
    expected: int,
) -> dict[str, object]:
    summary_path = artifact_dir / "profile_export_aiperf.json"
    rows_path = artifact_dir / "profile_export.jsonl"
    if not summary_path.is_file() or not rows_path.is_file():
        raise AIPerfValidationError(
            "AIPerf did not produce profile_export_aiperf.json and "
            "profile_export.jsonl"
        )
    payload = json.loads(summary_path.read_text())
    completed = _metric_avg(payload, "completed_request_count")
    requested = _metric_avg(payload, "request_count")
    error_rate = _metric_avg(payload, "request_error_rate")

    rows = [
        json.loads(line)
        for line in rows_path.read_text().splitlines()
        if line.strip()
    ]
    latency_ms: list[float] = []
    ttft_ms: list[float] = []
    starts: list[int] = []
    ends: list[int] = []
    successful_rows = 0
    for index, row in enumerate(rows):
        metadata = row.get("metadata")
        metrics = row.get("metrics")
        if isinstance(metadata, dict) and isinstance(metrics, dict):
            start_value = metadata.get("request_start_ns")
            end_value = metadata.get("request_end_ns")
            row_failed = (
                row.get("error") not in (None, "", {}, [])
                or metadata.get("was_cancelled") is True
                or metadata.get("context_overflow_skip") is True
            )
            ttft_metric = metrics.get("time_to_first_token")
            ttft_value = (
                ttft_metric.get("value")
                if isinstance(ttft_metric, dict)
                else None
            )
        else:
            row_failed = row.get("error") not in (None, "", {}, [])
            start_value = row.get("request_start_ns")
            end_value = row.get("request_end_ns")
            first_token = row.get("first_token_ns")
            ttft_value = (
                (int(first_token) - int(start_value)) / 1_000_000
                if first_token is not None and start_value is not None
                else None
            )
        if start_value is None or end_value is None:
            raise AIPerfValidationError(
                f"profiling row {index} is missing request timestamps"
            )
        start = int(start_value)
        end = int(end_value)
        if end < start:
            raise AIPerfValidationError(
                f"profiling row {index} ends before it starts"
            )
        starts.append(start)
        ends.append(end)
        if not row_failed:
            if ttft_value is None:
                raise AIPerfValidationError(
                    f"profiling row {index} is missing time_to_first_token"
                )
            successful_rows += 1
            latency_ms.append((end - start) / 1_000_000)
            ttft_ms.append(float(ttft_value))
    dispatch_window = (
        (max(starts) - min(starts)) / 1_000_000_000 if len(starts) > 1 else 0.0
    )
    request_window = (
        (max(ends) - min(starts)) / 1_000_000_000 if starts else 0.0
    )
    error_summary = payload.get("error_summary")
    summary: dict[str, object] = {
        "expected": expected,
        "recorded": len(rows),
        "completed": int(completed),
        "successful": int(requested),
        "failed": int(completed - requested),
        "error_rate_pct": error_rate,
        "errors": {} if error_summary in (None, {}, [], "") else {
            "aiperf": error_summary
        },
        "error_summary": error_summary,
        "is_complete": False,
        "strict_completion": False,
        "dispatch_window_seconds": dispatch_window,
        "dispatch_rps": len(rows) / dispatch_window if dispatch_window > 0 else None,
        "request_window_seconds": request_window,
        "latency_ms_successful": _percentiles(latency_ms),
        "ttft_ms_successful": _percentiles(ttft_ms),
        "aiperf_summary": payload,
    }

    def fail(message: str) -> None:
        raise AIPerfValidationError(message, summary)

    if payload.get("is_complete") is not True:
        fail("AIPerf marked the profile incomplete")
    if completed != expected or requested != expected:
        fail(
            f"AIPerf completion mismatch: expected={expected}, "
            f"completed={completed}, requested={requested}"
        )
    if len(rows) != expected:
        fail(f"expected {expected} profiling rows, observed {len(rows)}")
    if successful_rows != int(requested):
        fail(
            "AIPerf successful-row mismatch: "
            f"summary={int(requested)}, rows={successful_rows}"
        )
    if error_rate != 0:
        fail(f"AIPerf request_error_rate={error_rate}")
    if error_summary not in (None, {}, [], ""):
        fail(f"AIPerf error_summary is non-empty: {error_summary!r}")
    if "osl_mismatch_count" in payload:
        if _metric_avg(payload, "osl_mismatch_count") != 0:
            fail("AIPerf reported output-length mismatches")

    summary["is_complete"] = True
    summary["strict_completion"] = True
    return summary


def run_aiperf(
    kubernetes: Kubernetes,
    baseline: FleetSnapshot,
    executable: Path,
    trace: Path,
    artifact_dir: Path,
    request_rate: float,
    expected_replicas: Mapping[str, int] = EXPECTED_REPLICAS,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    request_count = trace_request_count(trace)
    dataset_entries = trace_dataset_entries(trace)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    command = aiperf_command(
        executable,
        trace,
        artifact_dir,
        request_rate,
        request_count,
        dataset_entries,
    )
    env = _aiperf_environment()
    timeout = max(1800.0, 960.0 + request_count / request_rate + 240.0)
    observations: list[dict[str, object]] = []

    def stop_process(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    with (artifact_dir / "stdout.log").open("w") as stdout, (
        artifact_dir / "stderr.log"
    ).open("w") as stderr:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        try:
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"AIPerf exceeded {timeout:.0f} seconds")
                snapshot = kubernetes.fleet()
                validate_fleet(snapshot, expected_replicas)
                stable = snapshot.stable_against(baseline)
                observations.append(
                    {
                        "timestamp": time.time(),
                        "stable": stable,
                        "fleet": snapshot.as_dict(),
                    }
                )
                if not stable:
                    raise AIPerfValidationError(
                        "fleet names, UIDs, readiness, roles, or restart counts "
                        "changed during traffic"
                    )
                time.sleep(FLEET_POLL_SECONDS)
        finally:
            stop_process(process)
        if process.returncode:
            raise AIPerfValidationError(
                f"AIPerf exited with status {process.returncode}; "
                f"see {artifact_dir / 'stderr.log'}"
            )
    return validate_outputs(artifact_dir, request_count), observations
