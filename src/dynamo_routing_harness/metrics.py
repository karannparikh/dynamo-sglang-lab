from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .constants import METRICS_TIMEOUT_SECONDS, RANKS_PER_PROCESS
from .kubernetes import FleetSnapshot

QUERY_METRIC = "sglang:simulated_prefix_cache_query_tokens_total"
HIT_METRIC = "sglang:simulated_prefix_cache_hit_tokens_total"
METRIC_FIELDS = {
    QUERY_METRIC: "query_tokens",
    HIT_METRIC: "hit_tokens",
}
KV_RETURN_METRIC_FIELDS = {
    "sglang:simulated_kv_return_attempts_total": "attempts",
    "sglang:simulated_kv_return_stored_total": "stored",
    "sglang:simulated_kv_return_rejected_total": "rejected",
    "sglang:simulated_kv_return_failed_total": "failed",
    "sglang:simulated_kv_return_tokens_total": "tokens",
    "sglang:simulated_kv_return_bytes_total": "bytes",
    "sglang:simulated_kv_return_transfer_seconds_total": "transfer_seconds",
    "sglang:simulated_kv_return_commit_seconds_total": "commit_seconds",
}
KV_RETURN_CORE_FIELDS = {
    "attempts",
    "stored",
    "tokens",
    "bytes",
    "transfer_seconds",
    "commit_seconds",
}


class MetricValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class CacheSnapshot:
    counters: dict[str, dict[str, float]]

    def identities(self) -> set[str]:
        return set(self.counters)

    def as_dict(self) -> dict[str, object]:
        return {"ranks": self.counters}


@dataclass(frozen=True)
class KVReturnSnapshot:
    totals: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {"totals": self.totals}


def _identity(labels: dict[str, str]) -> str:
    return "|".join((labels["pod"], labels["worker_id"], labels["dp_rank"]))


def _prometheus_query(url: str, query: str) -> list[dict[str, object]]:
    endpoint = f"{url.rstrip('/')}/api/v1/query?{urllib.parse.urlencode({'query': query})}"
    with urllib.request.urlopen(endpoint, timeout=10) as response:
        payload = json.load(response)
    if payload.get("status") != "success":
        raise MetricValidationError(f"Prometheus query failed: {payload!r}")
    return payload.get("data", {}).get("result", [])


def _promql_regex_string(values: list[str]) -> str:
    pattern = "|".join(
        re.escape(value).replace(r"\-", "-")
        for value in values
    )
    return pattern.replace("\\", "\\\\").replace('"', '\\"')


def read_cache_snapshot(
    prometheus_url: str,
    fleet: FleetSnapshot,
) -> CacheSnapshot:
    pods = sorted(pod.name for pod in fleet.by_role("prefill"))
    if not pods:
        raise MetricValidationError("no prefill pods are present")
    pod_pattern = _promql_regex_string(pods)
    query = (
        '{__name__=~"sglang:simulated_prefix_cache_(query|hit)_tokens_total",'
        f'pod=~"{pod_pattern}"'
        "}"
    )
    result = _prometheus_query(prometheus_url, query)
    counters: dict[str, dict[str, float]] = {}
    for series in result:
        labels = series.get("metric", {})
        if not isinstance(labels, dict):
            continue
        required = {"__name__", "pod", "dynamo_component_type", "worker_id", "dp_rank"}
        if not required.issubset(labels):
            raise MetricValidationError(
                f"cache metric is missing labels {sorted(required - set(labels))}: "
                f"{labels!r}"
            )
        if labels["pod"] not in pods:
            raise MetricValidationError(
                f"cache metric escaped current pod scope: {labels['pod']}"
            )
        if labels["dynamo_component_type"] != "prefill":
            raise MetricValidationError(
                "cache metric has unexpected component type: "
                f"{labels['dynamo_component_type']!r}"
            )
        field = METRIC_FIELDS.get(labels["__name__"])
        if field is None:
            continue
        value = series.get("value", [])
        if not isinstance(value, list) or len(value) != 2:
            raise MetricValidationError(f"invalid Prometheus value: {value!r}")
        counters.setdefault(_identity(labels), {})[field] = float(value[1])
    return CacheSnapshot(counters)


def validate_cache_snapshot(snapshot: CacheSnapshot, fleet: FleetSnapshot) -> None:
    expected = len(fleet.by_role("prefill")) * RANKS_PER_PROCESS
    if len(snapshot.counters) != expected:
        raise MetricValidationError(
            f"expected cache counters for {expected} prefill ranks, "
            f"observed {len(snapshot.counters)}"
        )
    for identity, counters in snapshot.counters.items():
        missing = {"query_tokens", "hit_tokens"} - set(counters)
        if missing:
            raise MetricValidationError(
                f"{identity} is missing cache counters: {sorted(missing)}"
            )


def wait_cache_snapshot(
    prometheus_url: str,
    fleet: FleetSnapshot,
    timeout: float = METRICS_TIMEOUT_SECONDS,
) -> CacheSnapshot:
    deadline = time.monotonic() + timeout
    last_error = "no metrics returned"
    while time.monotonic() < deadline:
        try:
            snapshot = read_cache_snapshot(prometheus_url, fleet)
            validate_cache_snapshot(snapshot, fleet)
            return snapshot
        except (MetricValidationError, OSError, ValueError) as error:
            last_error = str(error)
            time.sleep(5)
    raise TimeoutError(f"cache metrics did not become complete: {last_error}")


def read_kv_return_snapshot(
    prometheus_url: str,
    fleet: FleetSnapshot,
) -> KVReturnSnapshot:
    pods = sorted(
        pod.name
        for role in ("prefill", "decode")
        for pod in fleet.by_role(role)
    )
    if not pods:
        raise MetricValidationError("no worker pods are present")
    pod_pattern = _promql_regex_string(pods)
    query = (
        '{__name__=~"sglang:simulated_kv_return_.*",'
        f'pod=~"{pod_pattern}"'
        "}"
    )
    totals = {field: 0.0 for field in KV_RETURN_METRIC_FIELDS.values()}
    observed: set[str] = set()
    for series in _prometheus_query(prometheus_url, query):
        labels = series.get("metric", {})
        if not isinstance(labels, dict):
            continue
        metric_name = labels.get("__name__")
        pod = labels.get("pod")
        if pod not in pods:
            raise MetricValidationError(
                f"KV return metric escaped current pod scope: {pod!r}"
            )
        field = KV_RETURN_METRIC_FIELDS.get(str(metric_name))
        if field is None:
            continue
        value = series.get("value", [])
        if not isinstance(value, list) or len(value) != 2:
            raise MetricValidationError(f"invalid Prometheus value: {value!r}")
        totals[field] += float(value[1])
        observed.add(field)
    missing = KV_RETURN_CORE_FIELDS - observed
    if missing:
        raise MetricValidationError(
            f"KV return metrics are missing core counters: {sorted(missing)}"
        )
    return KVReturnSnapshot(totals)


def wait_kv_return_snapshot(
    prometheus_url: str,
    fleet: FleetSnapshot,
    timeout: float = METRICS_TIMEOUT_SECONDS,
) -> KVReturnSnapshot:
    deadline = time.monotonic() + timeout
    last_error = "no metrics returned"
    while time.monotonic() < deadline:
        try:
            return read_kv_return_snapshot(prometheus_url, fleet)
        except (MetricValidationError, OSError, ValueError) as error:
            last_error = str(error)
            time.sleep(5)
    raise TimeoutError(f"KV return metrics did not become complete: {last_error}")


def cache_delta(
    before: CacheSnapshot,
    after: CacheSnapshot,
    *,
    fleet_stable: bool,
) -> dict[str, object]:
    if before.identities() != after.identities():
        raise MetricValidationError(
            "cache metric identities changed between direct baseline and final scrape"
        )
    hit_delta = 0.0
    query_delta = 0.0
    per_rank: dict[str, dict[str, float]] = {}
    for identity in sorted(before.identities()):
        rank_delta = {
            field: after.counters[identity][field] - before.counters[identity][field]
            for field in ("hit_tokens", "query_tokens")
        }
        if any(value < 0 for value in rank_delta.values()):
            raise MetricValidationError(
                f"counter reset detected for current rank {identity}: {rank_delta!r}"
            )
        per_rank[identity] = rank_delta
        hit_delta += rank_delta["hit_tokens"]
        query_delta += rank_delta["query_tokens"]
    if query_delta <= 0:
        raise MetricValidationError(
            "query_tokens delta must be positive for the current case"
        )
    return {
        "fleet_stable": fleet_stable,
        "prefill_ranks_observed": len(before.counters),
        "prefill_ranks_expected": len(before.counters),
        "rank_coverage_complete": True,
        "hit_tokens_delta": hit_delta,
        "query_tokens_delta": query_delta,
        "token_hit_fraction": hit_delta / query_delta if fleet_stable else None,
        "scope": (
            "direct before/after counter deltas from the newly recreated "
            "prefill pod identities"
        ),
        "per_rank": per_rank,
    }


def kv_return_delta(
    before: KVReturnSnapshot,
    after: KVReturnSnapshot,
) -> dict[str, float]:
    delta = {
        field: after.totals[field] - before.totals[field]
        for field in before.totals
    }
    if any(value < 0 for value in delta.values()):
        raise MetricValidationError(
            f"KV return counter reset detected: {delta!r}"
        )
    return delta


def write_cache_snapshot(snapshot: CacheSnapshot, output: Path) -> None:
    output.write_text(json.dumps(snapshot.as_dict(), indent=2, sort_keys=True) + "\n")


def write_kv_return_snapshot(snapshot: KVReturnSnapshot, output: Path) -> None:
    output.write_text(json.dumps(snapshot.as_dict(), indent=2, sort_keys=True) + "\n")
