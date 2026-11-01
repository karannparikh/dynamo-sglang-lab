from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .constants import expected_cases, topology as default_topology


class ReportValidationError(RuntimeError):
    pass


def _load_cases(
    output: Path,
    expected_cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for expected in expected_cases:
        path = output / expected["case"] / "summary.json"
        if not path.is_file():
            raise ReportValidationError(f"missing case summary: {path}")
        payload = json.loads(path.read_text())
        for field in ("workload", "routing_cli_mode", "routing_label", "case"):
            if payload.get(field) != expected[field]:
                raise ReportValidationError(
                    f"{path} has invalid {field}: {payload.get(field)!r}"
                )
        if payload.get("kv_return_enabled", False) != expected.get(
            "kv_return_enabled",
            False,
        ):
            raise ReportValidationError(
                f"{path} has invalid kv_return_enabled: "
                f"{payload.get('kv_return_enabled')!r}"
            )
        cases.append(payload)
    return cases


def _value(payload: dict[str, Any], *path: str) -> Any:
    current: Any = payload
    for item in path:
        if not isinstance(current, dict):
            return None
        current = current.get(item)
    return current


def _number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def build_markdown(summary: dict[str, Any]) -> str:
    sources = summary["sources"]
    topology = summary.get("topology") or default_topology()
    workloads = summary.get("workloads") or list(
        dict.fromkeys(case["workload"] for case in summary["cases"])
    )
    routing_modes = summary.get("routing_modes") or list(
        dict.fromkeys(
            case.get("routing_cli_mode", case["routing_label"])
            for case in summary["cases"]
        )
    )
    lines = [
        "# Dynamo routing benchmark",
        "",
        f"- Offered RPS: `{summary['offered_rps']}`",
        f"- Workloads: {', '.join(f'`{workload}`' for workload in workloads)}",
        f"- Routing: {', '.join(f'`{mode}`' for mode in routing_modes)}",
        (
            f"- Topology: {topology['frontend_processes']} frontend, "
            f"{topology['planner_processes']} Planner, "
            f"{topology['prefill_processes']} prefill processes x "
            f"{topology['ranks_per_worker_process']} ranks, "
            f"{topology['decode_processes']} decode processes x "
            f"{topology['ranks_per_worker_process']} ranks"
        ),
        (
            "- Router replica synchronization: "
            f"`{'enabled' if summary.get('router_replica_sync') else 'disabled'}`"
        ),
        "- H200 profile: 15,509 blocks/rank, 256 tokens/page, "
        "3,970,304 slots/rank, 1,048,576 context, FP8 E4M3, "
        "32 running requests/rank",
    ]
    output_token_limit = summary.get("output_token_limit")
    if output_token_limit is not None:
        lines.append(
            f"- Requested output-token cap: `{output_token_limit}` tokens "
            "(trace lengths above the cap are clamped; shorter requests are unchanged)"
        )
    fake_output_token_id = summary.get("fake_output_token_id")
    if fake_output_token_id is not None:
        lines.append(
            f"- Fixed round-trip-safe simulated output token ID: "
            f"`{fake_output_token_id}`"
        )
    lines.extend(
        [
            "",
            "## Local sources",
            "",
            "| Source | HEAD | Requested | Ref | Dirty | Path |",
            "|---|---|---|---|---:|---|",
        ]
    )
    for name in ("dynamo", "fake_sglang"):
        source = sources[name]
        lines.append(
            f"| {name.replace('_', '-')} | `{source['head']}` | "
            f"`{source.get('requested') or 'n/a'}` | "
            f"`{source.get('ref') or 'n/a'}` | "
            f"{'yes' if source['dirty'] else 'no'} | `{source['path']}` |"
        )
    trace_evidence = summary.get("trace_evidence")
    if isinstance(trace_evidence, dict):
        lines.extend(
            [
                "",
                "## Request trace evidence",
                "",
                "| Workload | Records | Sessions | Turns/session | Input tokens | "
                "Capped | Original output max | Effective output max | "
                "Effective SHA-256 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for workload in workloads:
            evidence = trace_evidence.get(workload)
            if not isinstance(evidence, dict):
                continue
            turns = evidence.get("turns_per_session", {})
            turn_text = (
                f"{_number(turns.get('mean'))} "
                f"({turns.get('min')}-{turns.get('max')})"
                if isinstance(turns, dict)
                else "n/a"
            )
            input_tokens = evidence.get("input_tokens", {})
            input_text = (
                f"{input_tokens.get('min')}-{input_tokens.get('max')}"
                if isinstance(input_tokens, dict)
                else "n/a"
            )
            lines.append(
                f"| {workload} | {evidence['records']} | "
                f"{evidence.get('sessions', 'n/a')} | {turn_text} | "
                f"{input_text} | {evidence['capped_records']} | "
                f"{_value(evidence, 'original_output_tokens', 'max')} | "
                f"{_value(evidence, 'effective_output_tokens', 'max')} | "
                f"`{evidence['effective_sha256']}` |"
            )
    lines.extend(
        [
            "",
            "## Matrix",
            "",
            "| Workload | Routing | KV return | Attempts | Stored | Rejected | "
            "Failed | Returned tokens | Successful | Errors | Dispatch RPS | "
            "Cache hit | Latency p99 ms | TTFT p99 ms | Stable |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
            "---:|---:|---:|",
        ]
    )
    for case in summary["cases"]:
        errors = case.get("failed")
        if errors is None:
            errors = len(case.get("errors", {}))
        hit_fraction = _value(case, "prefill_cache", "token_hit_fraction")
        hit = "n/a" if hit_fraction is None else f"{100 * float(hit_fraction):.2f}%"
        lines.append(
            "| {workload} | {routing} | {kv_return} | {attempts} | {stored} | "
            "{rejected} | {failed_returns} | {returned_tokens} | {successful} | "
            "{errors} | {rps} | {hit} | {latency} | {ttft} | {stable} |".format(
                workload=case["workload"],
                routing=case["routing_label"],
                kv_return="on" if case.get("kv_return_enabled") else "off",
                attempts=_number(_value(case, "kv_return", "attempts"), 0),
                stored=_number(_value(case, "kv_return", "stored"), 0),
                rejected=_number(_value(case, "kv_return", "rejected"), 0),
                failed_returns=_number(
                    _value(case, "kv_return", "failed"),
                    0,
                ),
                returned_tokens=_number(
                    _value(case, "kv_return", "tokens"),
                    0,
                ),
                successful=case.get("successful", 0),
                errors=errors,
                rps=_number(case.get("dispatch_rps")),
                hit=hit,
                latency=_number(
                    _value(case, "latency_ms_successful", "p99")
                ),
                ttft=_number(_value(case, "ttft_ms_successful", "p99")),
                stable="yes" if case.get("fleet_stable") else "no",
            )
        )
    return "\n".join(lines) + "\n"


def generate_report(output: Path) -> dict[str, Any]:
    matrix_path = output / "matrix.json"
    if not matrix_path.is_file():
        raise ReportValidationError(f"missing benchmark matrix: {matrix_path}")
    matrix = json.loads(matrix_path.read_text())
    expected = matrix.get("expected_cases") or expected_cases()
    cases = _load_cases(output, expected)
    matrix_complete = (
        matrix.get("cases_finished") is True
        and matrix.get("comparison_complete") is True
        and len(cases) == len(expected)
        and all(case.get("strict_completion") is True for case in cases)
    )
    summary = {
        "matrix_complete": matrix_complete,
        "offered_rps": matrix["offered_rps"],
        "workloads": matrix.get("workloads"),
        "routing_modes": matrix.get("routing_modes"),
        "kv_return_modes": matrix.get("kv_return_modes"),
        "fake_output_token_id": matrix.get("fake_output_token_id"),
        "router_replica_sync": matrix.get("router_replica_sync", False),
        "output_token_limit": matrix.get("output_token_limit"),
        "trace_evidence": matrix.get("trace_evidence"),
        "h200_profile_from_application_defaults": matrix[
            "h200_profile_from_application_defaults"
        ],
        "topology": matrix["topology"],
        "runtime_image": matrix["runtime_image"],
        "planner_image": matrix["planner_image"],
        "sources": matrix["sources"],
        "expected_cases": expected,
        "cases": cases,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (output / "report.md").write_text(build_markdown(summary))
    return summary
