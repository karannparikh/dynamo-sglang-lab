from __future__ import annotations

import unittest
from unittest.mock import patch

from dynamo_routing_harness.kubernetes import parse_fleet
from dynamo_routing_harness.metrics import (
    CacheSnapshot,
    HIT_METRIC,
    KVReturnSnapshot,
    QUERY_METRIC,
    _promql_regex_string,
    cache_delta,
    kv_return_delta,
    read_cache_snapshot,
    validate_cache_snapshot,
)
from test_kubernetes import complete_payload


def series(fleet):
    result = []
    for pod in fleet.by_role("prefill"):
        for rank in range(10):
            for metric, value in ((QUERY_METRIC, 100), (HIT_METRIC, 40)):
                result.append(
                    {
                        "metric": {
                            "__name__": metric,
                            "pod": pod.name,
                            "dynamo_component_type": "prefill",
                            "worker_id": pod.name,
                            "dp_rank": str(rank),
                        },
                        "value": [1, str(value)],
                    }
                )
    return result


class MetricsTests(unittest.TestCase):
    def test_kv_return_counter_delta(self) -> None:
        before = KVReturnSnapshot(
            {
                "attempts": 1,
                "stored": 1,
                "rejected": 0,
                "failed": 0,
                "tokens": 10,
                "bytes": 20,
                "transfer_seconds": 0.1,
                "commit_seconds": 0.01,
            }
        )
        after = KVReturnSnapshot(
            {
                "attempts": 4,
                "stored": 3,
                "rejected": 1,
                "failed": 0,
                "tokens": 30,
                "bytes": 60,
                "transfer_seconds": 0.3,
                "commit_seconds": 0.03,
            }
        )
        self.assertEqual(kv_return_delta(before, after)["stored"], 2)

    def test_promql_regex_string_escapes_for_prometheus(self) -> None:
        pattern = _promql_regex_string(["prefill-a", "prefill.b"])
        self.assertEqual(pattern, r"prefill-a|prefill\\.b")
        self.assertNotIn(r"\-", pattern)

    def test_complete_snapshot_has_40_ranks(self) -> None:
        fleet = parse_fleet(complete_payload())
        with patch(
            "dynamo_routing_harness.metrics._prometheus_query",
            return_value=series(fleet),
        ):
            snapshot = read_cache_snapshot("http://prometheus", fleet)
        validate_cache_snapshot(snapshot, fleet)
        self.assertEqual(len(snapshot.counters), 40)

    def test_direct_delta_and_hit_fraction(self) -> None:
        before = CacheSnapshot(
            {"pod|worker|0": {"query_tokens": 10, "hit_tokens": 2}}
        )
        after = CacheSnapshot(
            {"pod|worker|0": {"query_tokens": 30, "hit_tokens": 12}}
        )
        delta = cache_delta(before, after, fleet_stable=True)
        self.assertEqual(delta["query_tokens_delta"], 20)
        self.assertEqual(delta["hit_tokens_delta"], 10)
        self.assertEqual(delta["token_hit_fraction"], 0.5)

    def test_unstable_fleet_suppresses_fraction(self) -> None:
        before = CacheSnapshot(
            {"pod|worker|0": {"query_tokens": 1, "hit_tokens": 0}}
        )
        after = CacheSnapshot(
            {"pod|worker|0": {"query_tokens": 2, "hit_tokens": 1}}
        )
        self.assertIsNone(
            cache_delta(before, after, fleet_stable=False)["token_hit_fraction"]
        )


if __name__ == "__main__":
    unittest.main()
