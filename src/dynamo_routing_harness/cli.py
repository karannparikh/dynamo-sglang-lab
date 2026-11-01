from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .aiperf import setup_aiperf
from .benchmark import run_matrix
from .build import ImageBuilder, image_names
from .constants import (
    DEFAULT_KUBECONFIG,
    DEFAULT_REQUEST_RATE,
    LOCAL_ROOT,
    replica_counts,
)
from .kubernetes import Kubernetes, validate_fleet
from .manifests import render_resources, write_json_documents
from .report import generate_report
from .source import inspect_sources, write_source_record
from .traces import ensure_traces, write_multiturn_trace


def _default_trace_dir() -> Path:
    return LOCAL_ROOT / "mooncake-100rps"


def _default_aiperf() -> Path:
    return LOCAL_ROOT / "aiperf-venv" / "bin" / "aiperf"


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="dynamo-routing-lab",
        description="Apple Silicon CPU-only local Dynamo routing harness",
    )
    root.add_argument(
        "--kubeconfig",
        type=Path,
        default=DEFAULT_KUBECONFIG,
    )
    root.add_argument(
        "--dynamo-source",
        help="local Git checkout or GitHub location; defaults to DYNAMO_SOURCE",
    )
    root.add_argument(
        "--dynamo-ref",
        help="branch, tag, or commit for a remote --dynamo-source",
    )
    root.add_argument(
        "--fake-sglang-source",
        help="local Git checkout or GitHub location; defaults to FAKE_SGLANG_SOURCE",
    )
    root.add_argument(
        "--fake-sglang-ref",
        help="branch, tag, or commit for a remote --fake-sglang-source",
    )
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    commands.add_parser("up")
    commands.add_parser("rebuild")
    deploy = commands.add_parser("deploy")
    deploy.add_argument(
        "--routing-mode",
        choices=("kv", "load-aware", "power-of-two"),
        default="kv",
    )
    deploy.add_argument("--enable-kv-return", action="store_true")
    deploy.add_argument("--frontend-replicas", type=int, default=1)
    deploy.add_argument(
        "--router-replica-sync",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    setup = commands.add_parser("benchmark-setup")
    setup.add_argument("trace_dir", nargs="?", type=Path, default=_default_trace_dir())
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument(
        "--request-rate",
        type=float,
        default=DEFAULT_REQUEST_RATE,
    )
    benchmark.add_argument(
        "--output-token-limit",
        type=int,
        help=(
            "cap each Mooncake trace row's output_length without increasing "
            "shorter requests"
        ),
    )
    benchmark.add_argument(
        "--workload",
        action="append",
        choices=("conversation", "toolagent"),
        dest="workloads",
    )
    benchmark.add_argument(
        "--routing-mode",
        action="append",
        choices=("kv", "load-aware", "power-of-two"),
        dest="routing_modes",
    )
    benchmark.add_argument(
        "--kv-return",
        choices=("off", "on", "both"),
        default="off",
    )
    benchmark.add_argument(
        "--trace-dir",
        type=Path,
        default=_default_trace_dir(),
    )
    benchmark.add_argument("--aiperf", type=Path, default=_default_aiperf())
    benchmark.add_argument("--frontend-replicas", type=int, default=1)
    benchmark.add_argument(
        "--router-replica-sync",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    multiturn = commands.add_parser("benchmark-multiturn")
    multiturn.add_argument("--output", type=Path, required=True)
    multiturn.add_argument(
        "--request-rate",
        type=float,
        default=DEFAULT_REQUEST_RATE,
    )
    multiturn.add_argument("--sessions", type=int, default=1500)
    multiturn.add_argument("--turns", type=int, default=8)
    multiturn.add_argument("--input-tokens", type=int, default=256)
    multiturn.add_argument("--output-tokens", type=int, default=256)
    multiturn.add_argument(
        "--kv-return",
        choices=("off", "on", "both"),
        default="both",
    )
    multiturn.add_argument("--aiperf", type=Path, default=_default_aiperf())
    multiturn.add_argument("--frontend-replicas", type=int, default=1)
    multiturn.add_argument(
        "--router-replica-sync",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    report = commands.add_parser("report")
    report.add_argument("--output", type=Path, required=True)
    render = commands.add_parser("render")
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--frontend-replicas", type=int, default=1)
    render.add_argument(
        "--router-replica-sync",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    commands.add_parser("status")
    commands.add_parser("stop")
    return root


def _sources_json(sources: object) -> str:
    return json.dumps(sources.as_dict(), indent=2, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    kubernetes = Kubernetes(kubeconfig=args.kubeconfig)
    source_environ = os.environ.copy()
    overrides = {
        "DYNAMO_SOURCE": args.dynamo_source,
        "DYNAMO_REF": args.dynamo_ref,
        "FAKE_SGLANG_SOURCE": args.fake_sglang_source,
        "FAKE_SGLANG_REF": args.fake_sglang_ref,
    }
    source_environ.update(
        {name: value for name, value in overrides.items() if value is not None}
    )

    if args.command == "doctor":
        kubernetes.require_apple_silicon()
        kubernetes.require_tools()
        sources = inspect_sources(environ=source_environ)
        print(_sources_json(sources))
        return 0
    if args.command == "stop":
        kubernetes.stop()
        return 0
    if args.command == "status":
        sys.stdout.write(kubernetes.status())
        return 0
    if args.command == "benchmark-setup":
        ensure_traces(args.trace_dir)
        executable = setup_aiperf()
        print(executable)
        return 0
    if args.command == "report":
        summary = generate_report(args.output)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["matrix_complete"] else 1

    sources = inspect_sources(environ=source_environ)
    images = image_names(sources)
    write_source_record(sources, LOCAL_ROOT / "sources.json")

    if args.command == "render":
        args.output.mkdir(parents=True, exist_ok=True)
        for routing_mode in ("kv", "load-aware", "power-of-two"):
            write_json_documents(
                render_resources(
                    images.serving,
                    images.planner,
                    sources,
                    routing_mode,
                    frontend_replicas=args.frontend_replicas,
                    router_replica_sync=args.router_replica_sync,
                ),
                args.output / f"dgd-{routing_mode}.json",
            )
        print(args.output)
        return 0
    if args.command == "up":
        kubernetes.start_cluster()
        images = ImageBuilder().build(sources)
        kubernetes.install_platform(sources, images)
        kubernetes.install_observability()
        kubernetes.deploy_graph(images.serving, images.planner, sources)
        kubernetes.wait_graph()
        fleet = kubernetes.wait_fleet()
        validate_fleet(fleet)
        kubernetes.validate_profile_contract(fleet)
        print(kubernetes.status(), end="")
        return 0
    if args.command == "rebuild":
        images = ImageBuilder().build(sources)
        kubernetes.install_platform(sources, images)
        kubernetes.deploy_graph(images.serving, images.planner, sources)
        kubernetes.wait_graph()
        print(json.dumps(images.__dict__, indent=2, sort_keys=True))
        return 0
    if args.command == "deploy":
        kubernetes.deploy_graph(
            images.serving,
            images.planner,
            sources,
            args.routing_mode,
            args.enable_kv_return,
            frontend_replicas=args.frontend_replicas,
            router_replica_sync=args.router_replica_sync,
        )
        kubernetes.wait_graph()
        replicas = replica_counts(args.frontend_replicas)
        fleet = kubernetes.wait_fleet(expected_replicas=replicas)
        validate_fleet(fleet, replicas)
        kubernetes.validate_profile_contract(fleet)
        return 0
    if args.command == "benchmark":
        kv_return_modes = {
            "off": (False,),
            "on": (True,),
            "both": (False, True),
        }[args.kv_return]
        report = run_matrix(
            kubernetes,
            sources,
            images,
            args.output,
            args.trace_dir,
            args.aiperf,
            args.request_rate,
            args.output_token_limit,
            tuple(dict.fromkeys(args.workloads or ("conversation", "toolagent"))),
            tuple(
                dict.fromkeys(
                    args.routing_modes or ("kv", "load-aware", "power-of-two")
                )
            ),
            kv_return_modes,
            frontend_replicas=args.frontend_replicas,
            router_replica_sync=args.router_replica_sync,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.command == "benchmark-multiturn":
        trace_dir = args.output / "source-traces"
        write_multiturn_trace(
            trace_dir,
            sessions=args.sessions,
            turns=args.turns,
            input_tokens=args.input_tokens,
            output_tokens=args.output_tokens,
        )
        kv_return_modes = {
            "off": (False,),
            "on": (True,),
            "both": (False, True),
        }[args.kv_return]
        report = run_matrix(
            kubernetes,
            sources,
            images,
            args.output,
            trace_dir,
            args.aiperf,
            args.request_rate,
            args.output_tokens,
            ("multiturn",),
            ("kv",),
            kv_return_modes,
            264,
            args.frontend_replicas,
            args.router_replica_sync,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
