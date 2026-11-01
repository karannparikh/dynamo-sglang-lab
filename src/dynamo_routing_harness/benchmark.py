from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

from .aiperf import AIPerfValidationError, run_aiperf
from .build import Images
from .constants import (
    DEFAULT_REQUEST_RATE,
    EXPECTED_REPLICAS,
    H200_PROFILE,
    RANKS_PER_PROCESS,
    ROUTING_MODES,
    SETTLE_SECONDS,
    SUPPORTED_WORKLOADS,
    WORKLOADS,
    case_name,
    replica_counts,
    topology,
)
from .kubernetes import Kubernetes, PortForward, validate_fleet
from .metrics import (
    cache_delta,
    kv_return_delta,
    wait_cache_snapshot,
    wait_kv_return_snapshot,
    write_cache_snapshot,
    write_kv_return_snapshot,
)
from .report import generate_report
from .source import SourcePair
from .traces import prepare_benchmark_traces


class BenchmarkFailed(RuntimeError):
    pass


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _case_base(
    case_name: str,
    workload: str,
    routing_mode: str,
    request_rate: float,
    sources: SourcePair,
    output_token_limit: int | None,
    trace_evidence: dict[str, object],
    enable_kv_return: bool,
    frontend_replicas: int,
    router_replica_sync: bool,
) -> dict[str, object]:
    replicas = replica_counts(frontend_replicas)
    return {
        "case": case_name,
        "workload": workload,
        "routing": ROUTING_MODES[routing_mode],
        "routing_label": ROUTING_MODES[routing_mode],
        "routing_cli_mode": routing_mode,
        "offered_rps": request_rate,
        "output_token_limit": output_token_limit,
        "trace_evidence": trace_evidence,
        "kv_return_enabled": enable_kv_return,
        "router_replica_sync": router_replica_sync,
        "sources": sources.as_dict(),
        "h200_profile_from_application_defaults": H200_PROFILE,
        "replicas": replicas,
        "ranks_per_process": {
            "prefill": RANKS_PER_PROCESS,
            "decode": RANKS_PER_PROCESS,
        },
        "logical_workers": {
            "prefill": replicas["prefill"] * RANKS_PER_PROCESS,
            "decode": replicas["decode"] * RANKS_PER_PROCESS,
        },
    }


def run_matrix(
    kubernetes: Kubernetes,
    sources: SourcePair,
    images: Images,
    output: Path,
    trace_dir: Path,
    aiperf: Path,
    request_rate: float = DEFAULT_REQUEST_RATE,
    output_token_limit: int | None = None,
    workloads: tuple[str, ...] = WORKLOADS,
    routing_modes: tuple[str, ...] = tuple(ROUTING_MODES),
    kv_return_modes: tuple[bool, ...] = (False,),
    fake_output_token_id: int | None = None,
    frontend_replicas: int = 1,
    router_replica_sync: bool = False,
) -> dict[str, object]:
    if request_rate <= 0:
        raise ValueError("request rate must be positive")
    if not workloads or any(
        workload not in SUPPORTED_WORKLOADS for workload in workloads
    ):
        raise ValueError(f"invalid workloads: {workloads!r}")
    if not routing_modes or any(mode not in ROUTING_MODES for mode in routing_modes):
        raise ValueError(f"invalid routing modes: {routing_modes!r}")
    if not kv_return_modes:
        raise ValueError("at least one KV return mode is required")
    replicas = replica_counts(frontend_replicas)
    if any(kv_return_modes) and tuple(routing_modes) != ("kv",):
        raise ValueError("KV return comparison requires KV-aware routing only")
    if not aiperf.is_file():
        raise FileNotFoundError(
            f"AIPerf executable not found: {aiperf}; run benchmark-setup first"
        )
    output.mkdir(parents=True, exist_ok=True)
    traces, trace_evidence = prepare_benchmark_traces(
        trace_dir,
        output / "input-traces",
        output_token_limit,
        workloads,
    )
    include_return_suffix = len(kv_return_modes) > 1 or any(kv_return_modes)
    selected_cases: list[dict[str, object]] = []
    for workload in workloads:
        for routing_mode in routing_modes:
            for enable_kv_return in kv_return_modes:
                base_name = case_name(workload, routing_mode)
                selected_cases.append(
                    {
                        "workload": workload,
                        "routing_cli_mode": routing_mode,
                        "routing_label": ROUTING_MODES[routing_mode],
                        "kv_return_enabled": enable_kv_return,
                        "case": (
                            f"{base_name}-kv-return-"
                            f"{'on' if enable_kv_return else 'off'}"
                            if include_return_suffix
                            else base_name
                        ),
                    }
                )
    matrix: dict[str, object] = {
        "offered_rps": request_rate,
        "workloads": list(workloads),
        "routing_modes": list(routing_modes),
        "kv_return_modes": list(kv_return_modes),
        "fake_output_token_id": fake_output_token_id,
        "router_replica_sync": router_replica_sync,
        "output_token_limit": output_token_limit,
        "trace_evidence": trace_evidence,
        "h200_profile_from_application_defaults": H200_PROFILE,
        "topology": topology(frontend_replicas),
        "runtime_image": images.serving,
        "planner_image": images.planner,
        "operator_image": images.operator,
        "sources": sources.as_dict(),
        "expected_cases": selected_cases,
        "cases_finished": False,
        "comparison_complete": False,
    }
    _write_json(output / "matrix.json", matrix)
    previous_uids: set[str] = set()
    failures: list[str] = []

    for selected in selected_cases:
            workload = str(selected["workload"])
            routing_mode = str(selected["routing_cli_mode"])
            enable_kv_return = bool(selected["kv_return_enabled"])
            current_case = str(selected["case"])
            case_dir = output / current_case
            case_dir.mkdir(parents=True, exist_ok=True)
            summary = _case_base(
                current_case,
                workload,
                routing_mode,
                request_rate,
                sources,
                output_token_limit,
                trace_evidence[workload],
                enable_kv_return,
                frontend_replicas,
                router_replica_sync,
            )
            try:
                kubernetes.deploy_graph(
                    images.serving,
                    images.planner,
                    sources,
                    routing_mode,
                    enable_kv_return,
                    fake_output_token_id,
                    frontend_replicas,
                    router_replica_sync,
                )
                kubernetes.wait_graph()
                current = kubernetes.fleet()
                previous_uids.update(current.uids)
                fresh = kubernetes.recreate_fleet(previous_uids, replicas)
                previous_uids.update(fresh.uids)
                time.sleep(SETTLE_SECONDS)
                baseline = kubernetes.fleet()
                validate_fleet(baseline, replicas)
                if not baseline.stable_against(fresh):
                    raise BenchmarkFailed(
                        "fleet changed during the mandatory 15-second settle"
                    )
                contract = kubernetes.validate_profile_contract(baseline)
                with PortForward(
                    kubernetes.kubeconfig,
                    "observability",
                    "service/prometheus",
                    "9090:9090",
                ):
                    before = wait_cache_snapshot(
                        "http://127.0.0.1:9090",
                        baseline,
                    )
                    kv_return_before = wait_kv_return_snapshot(
                        "http://127.0.0.1:9090",
                        baseline,
                    )
                write_cache_snapshot(
                    before,
                    case_dir / "cache-metrics-before.json",
                )
                write_kv_return_snapshot(
                    kv_return_before,
                    case_dir / "kv-return-metrics-before.json",
                )
                aiperf_summary, observations = run_aiperf(
                    kubernetes,
                    baseline,
                    aiperf,
                    traces[workload],
                    case_dir / "aiperf",
                    request_rate,
                    replicas,
                )
                final = kubernetes.fleet()
                validate_fleet(final, replicas)
                fleet_stable = final.stable_against(baseline) and all(
                    observation["stable"] for observation in observations
                )
                if not fleet_stable:
                    raise BenchmarkFailed(
                        "fleet changed after traffic completed"
                    )
                with PortForward(
                    kubernetes.kubeconfig,
                    "observability",
                    "service/prometheus",
                    "9090:9090",
                ):
                    after = wait_cache_snapshot(
                        "http://127.0.0.1:9090",
                        final,
                    )
                    kv_return_after = wait_kv_return_snapshot(
                        "http://127.0.0.1:9090",
                        final,
                    )
                write_cache_snapshot(
                    after,
                    case_dir / "cache-metrics-after.json",
                )
                write_kv_return_snapshot(
                    kv_return_after,
                    case_dir / "kv-return-metrics-after.json",
                )
                return_metrics = kv_return_delta(
                    kv_return_before,
                    kv_return_after,
                )
                if enable_kv_return and return_metrics["stored"] <= 0:
                    raise BenchmarkFailed(
                        "KV return was enabled but no response KV was stored"
                    )
                if not enable_kv_return and (
                    return_metrics["attempts"] > 0
                    or return_metrics["stored"] > 0
                ):
                    raise BenchmarkFailed(
                        "KV return counters advanced while the feature was disabled"
                    )
                summary.update(aiperf_summary)
                summary.update(
                    {
                        "fleet_stable": True,
                        "fleet": {
                            "baseline": baseline.as_dict(),
                            "final": final.as_dict(),
                            "observations": observations,
                        },
                        "effective_profile_contract": contract,
                        "kv_return": return_metrics,
                        "prefill_cache": cache_delta(
                            before,
                            after,
                            fleet_stable=True,
                        )
                    }
                )
            except Exception as error:
                failures.append(f"{current_case}: {error}")
                partial = (
                    error.summary
                    if isinstance(error, AIPerfValidationError)
                    else None
                )
                if partial is not None:
                    summary.update(partial)
                errors = dict(summary.get("errors", {}))
                errors["harness"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
                summary.update(
                    {
                        "is_complete": False,
                        "strict_completion": False,
                        "fleet_stable": False,
                        "successful": summary.get("successful", 0),
                        "errors": errors,
                        "aiperf_error_summary": summary.get("error_summary"),
                        "error_summary": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )
            _write_json(case_dir / "summary.json", summary)

    matrix["cases_finished"] = True
    matrix["comparison_complete"] = not failures
    _write_json(output / "matrix.json", matrix)
    report = generate_report(output)
    if failures:
        raise BenchmarkFailed("; ".join(failures))
    return report
