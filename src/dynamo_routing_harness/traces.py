from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Callable

from .constants import TRACE_SOURCES, WORKLOADS

Downloader = Callable[[str], bytes]


class TraceDownloadError(RuntimeError):
    pass


def _http_get(url: str) -> bytes:
    with urllib.request.urlopen(url) as response:  # noqa: S310 - pinned HTTPS URL
        return response.read()


def trace_path(trace_dir: Path, workload: str) -> Path:
    return trace_dir / f"{workload}_trace.jsonl"


def write_multiturn_trace(
    trace_dir: Path,
    *,
    sessions: int,
    turns: int,
    input_tokens: int,
    output_tokens: int,
) -> Path:
    values = {
        "sessions": sessions,
        "turns": turns,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    invalid = {
        name: value
        for name, value in values.items()
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0
    }
    if invalid:
        raise ValueError(f"multi-turn trace values must be positive integers: {invalid}")
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_path(trace_dir, "multiturn")
    with path.open("w") as output:
        for session_index in range(sessions):
            session_id = f"session-{session_index:06d}"
            for turn_index in range(turns):
                record = {
                    "session_id": session_id,
                    "input_length": input_tokens,
                    "output_length": output_tokens,
                }
                if turn_index:
                    record["delay"] = 0
                output.write(json.dumps(record, separators=(",", ":")) + "\n")
    return path


def _sha256(data: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


def _verify(workload: str, path: Path, expected_sha256: str) -> None:
    actual = _sha256(path.read_bytes())
    if actual != expected_sha256:
        raise TraceDownloadError(
            f"{workload} trace at {path} failed checksum: expected "
            f"{expected_sha256}, observed {actual}"
        )


def ensure_traces(
    trace_dir: Path,
    workloads: tuple[str, ...] = WORKLOADS,
    downloader: Downloader = _http_get,
) -> dict[str, Path]:
    """Ensure every configured Mooncake trace exists in ``trace_dir``.

    Existing files that already match their pinned checksum are left untouched.
    Missing traces are downloaded from their configured source and verified.
    Workloads without a configured source are left to the caller to provide.
    """
    trace_dir.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, Path] = {}
    for workload in workloads:
        path = trace_path(trace_dir, workload)
        source = TRACE_SOURCES.get(workload)
        if source is None:
            if not path.is_file():
                raise TraceDownloadError(
                    f"no download source configured for workload {workload!r}; "
                    f"place {path.name} in {trace_dir}"
                )
            resolved[workload] = path
            continue
        expected = source["sha256"]
        if path.is_file() and _sha256(path.read_bytes()) == expected:
            resolved[workload] = path
            continue
        data = downloader(source["url"])
        actual = _sha256(data)
        if actual != expected:
            raise TraceDownloadError(
                f"{workload} trace from {source['url']} failed checksum: "
                f"expected {expected}, observed {actual}"
            )
        path.write_bytes(data)
        _verify(workload, path, expected)
        resolved[workload] = path
    return resolved


def prepare_benchmark_traces(
    trace_dir: Path,
    output_dir: Path,
    output_token_limit: int | None,
    workloads: tuple[str, ...] = WORKLOADS,
) -> tuple[dict[str, Path], dict[str, dict[str, object]]]:
    if output_token_limit is not None and output_token_limit <= 0:
        raise ValueError("output token limit must be positive")

    resolved: dict[str, Path] = {}
    evidence: dict[str, dict[str, object]] = {}
    if output_token_limit is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for workload in workloads:
        source = trace_path(trace_dir, workload)
        if not source.is_file():
            raise FileNotFoundError(f"missing Mooncake trace: {source}")

        records: list[str] = []
        original_lengths: list[int] = []
        effective_lengths: list[int] = []
        input_lengths: list[int] = []
        session_turns: dict[str, int] = {}
        capped_records = 0
        for line_number, line in enumerate(source.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise TraceDownloadError(
                    f"{source}:{line_number} is not valid JSON: {error}"
                ) from error
            output_length = record.get("output_length")
            input_length = record.get("input_length")
            if (
                isinstance(output_length, bool)
                or not isinstance(output_length, int)
                or output_length <= 0
            ):
                raise TraceDownloadError(
                    f"{source}:{line_number} has invalid output_length "
                    f"{output_length!r}"
                )
            if (
                isinstance(input_length, bool)
                or not isinstance(input_length, int)
                or input_length <= 0
            ):
                raise TraceDownloadError(
                    f"{source}:{line_number} has invalid input_length "
                    f"{input_length!r}"
                )
            effective_length = (
                min(output_length, output_token_limit)
                if output_token_limit is not None
                else output_length
            )
            if effective_length != output_length:
                capped_records += 1
            record["output_length"] = effective_length
            records.append(json.dumps(record, separators=(",", ":")))
            original_lengths.append(output_length)
            effective_lengths.append(effective_length)
            input_lengths.append(input_length)
            session_id = record.get("session_id")
            identity = (
                session_id
                if isinstance(session_id, str) and session_id
                else f"row-{line_number}"
            )
            session_turns[identity] = session_turns.get(identity, 0) + 1

        if not records:
            raise TraceDownloadError(f"trace has no requests: {source}")

        if output_token_limit is None:
            effective = source
            effective_sha256 = _sha256(source.read_bytes())
        else:
            effective = output_dir / source.name
            effective.write_text("\n".join(records) + "\n")
            effective_sha256 = _sha256(effective.read_bytes())

        resolved[workload] = effective
        evidence[workload] = {
            "source_path": str(source),
            "effective_path": str(effective),
            "source_sha256": _sha256(source.read_bytes()),
            "effective_sha256": effective_sha256,
            "records": len(records),
            "sessions": len(session_turns),
            "turns_per_session": {
                "min": min(session_turns.values()),
                "max": max(session_turns.values()),
                "mean": len(records) / len(session_turns),
            },
            "input_tokens": {
                "min": min(input_lengths),
                "max": max(input_lengths),
            },
            "capped_records": capped_records,
            "output_token_limit": output_token_limit,
            "original_output_tokens": {
                "min": min(original_lengths),
                "max": max(original_lengths),
            },
            "effective_output_tokens": {
                "min": min(effective_lengths),
                "max": max(effective_lengths),
            },
        }

    return resolved, evidence
