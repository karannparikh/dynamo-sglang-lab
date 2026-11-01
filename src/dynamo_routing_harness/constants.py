from __future__ import annotations

from pathlib import Path

CLUSTER_NAME = "dynamo-routing"
COLIMA_PROFILE = "dynamo-routing"
KUBE_CONTEXT = f"kind-{CLUSTER_NAME}"
SERVING_NAMESPACE = "dynamo-sim"
OBSERVABILITY_NAMESPACE = "observability"
SYSTEM_NAMESPACE = "dynamo-system"
DGD_NAME = "fake-sglang"
MODEL_NAME = "fake-model"

# Vanilla (prebuilt) Dynamo release used by default. The harness pulls the
# published NGC operator/planner images and installs the ai-dynamo wheel for the
# serving runtime instead of compiling Dynamo from source. A local Dynamo source
# (explicit --dynamo-source / DYNAMO_SOURCE) switches back to the from-source
# build. The vanilla git tag is still checked out (shallow, never compiled) to
# stage the platform Helm chart.
DYNAMO_VERSION = "1.6.0"
DYNAMO_VANILLA_REPO = "ai-dynamo/dynamo"
DYNAMO_VANILLA_REF = f"v{DYNAMO_VERSION}"
DYNAMO_OPERATOR_IMAGE = (
    f"nvcr.io/nvidia/ai-dynamo/kubernetes-operator:{DYNAMO_VERSION}"
)
DYNAMO_PLANNER_IMAGE = f"nvcr.io/nvidia/ai-dynamo/dynamo-planner:{DYNAMO_VERSION}"

DYNAMO_RUNTIME_VERSION = DYNAMO_VERSION

# fake-sglang (the serving engine under test) is always built from source, but
# defaults to the canonical fork so the harness is self-contained. A local
# FAKE_SGLANG_SOURCE overrides this for development.
FAKE_SGLANG_VANILLA_REPO = "karannparikh/fake-sglang"
FAKE_SGLANG_VANILLA_REF = "main"
MODEL_PATH = "/opt/fake-sglang/models/qwen3-0.6b"
TOKENIZER = "Qwen/Qwen3-0.6B"
TOKENIZER_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"

DEFAULT_REQUEST_RATE = 20.0
SETTLE_SECONDS = 15
FLEET_POLL_SECONDS = 5
METRICS_TIMEOUT_SECONDS = 90

EXPECTED_REPLICAS = {
    "frontend": 1,
    "planner": 1,
    "prefill": 4,
    "decode": 4,
}
RANKS_PER_PROCESS = 10

H200_PROFILE = {
    "kv_blocks_per_rank": 15_509,
    "tokens_per_page": 256,
    "token_slots_per_rank": 3_970_304,
    "context_length": 1_048_576,
    "kv_cache_dtype": "fp8_e4m3",
    "max_running_requests_per_rank": 32,
}

ROUTING_MODES = {
    "kv": "KV-aware",
    "load-aware": "load-aware",
    "power-of-two": "PO2C",
}
WORKLOADS = ("conversation", "toolagent")
SUPPORTED_WORKLOADS = (*WORKLOADS, "multiturn")

# Mooncake FAST'25 traces (kvcache-ai/Mooncake), pinned for byte-reproducible,
# hash-verified downloads. AIPerf consumes these as a custom `mooncake_trace`
# dataset and does not fetch them itself, so the harness downloads them on
# demand into the (git-ignored) trace directory when they are missing.
MOONCAKE_TRACE_COMMIT = "3cca71daccf2a7afb8fe3f0295358f70e3a69fdb"
_MOONCAKE_TRACE_BASE_URL = (
    "https://raw.githubusercontent.com/kvcache-ai/Mooncake/"
    f"{MOONCAKE_TRACE_COMMIT}/FAST25-release/traces"
)
TRACE_SOURCES = {
    "conversation": {
        "url": f"{_MOONCAKE_TRACE_BASE_URL}/conversation_trace.jsonl",
        "sha256": (
            "b8cbb061a85206d729d91cdc2981f43c9e0d99209dce588d3af5f7934408b9df"
        ),
    },
    "toolagent": {
        "url": f"{_MOONCAKE_TRACE_BASE_URL}/toolagent_trace.jsonl",
        "sha256": (
            "48a2db1a13d3bc05e6330140c64f604ba366df20d3c9e128b5c35a01c1fa5f71"
        ),
    },
}

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LOCAL_ROOT = REPOSITORY_ROOT / ".local"
DEFAULT_KUBECONFIG = LOCAL_ROOT / "kubeconfig"


def case_name(workload: str, routing_mode: str) -> str:
    suffix = "po2c" if routing_mode == "power-of-two" else routing_mode
    return f"{workload}-{suffix}"


def expected_cases() -> list[dict[str, str]]:
    return [
        {
            "workload": workload,
            "routing_cli_mode": mode,
            "routing_label": label,
            "case": case_name(workload, mode),
        }
        for workload in WORKLOADS
        for mode, label in ROUTING_MODES.items()
    ]


def replica_counts(
    frontend_processes: int = EXPECTED_REPLICAS["frontend"],
) -> dict[str, int]:
    if frontend_processes <= 0:
        raise ValueError("frontend processes must be positive")
    return {**EXPECTED_REPLICAS, "frontend": frontend_processes}


def topology(
    frontend_processes: int = EXPECTED_REPLICAS["frontend"],
) -> dict[str, int]:
    replicas = replica_counts(frontend_processes)
    return {
        "frontend_processes": replicas["frontend"],
        "planner_processes": replicas["planner"],
        "prefill_processes": replicas["prefill"],
        "decode_processes": replicas["decode"],
        "ranks_per_worker_process": RANKS_PER_PROCESS,
    }
