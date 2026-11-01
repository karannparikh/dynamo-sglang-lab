from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

from dynamo_routing_harness.constants import H200_PROFILE
from dynamo_routing_harness.manifests import (
    render_grafana_dashboard,
    render_resources,
)
from dynamo_routing_harness.source import SourceMetadata, SourcePair


def sources() -> SourcePair:
    return SourcePair(
        dynamo=SourceMetadata("/dynamo", "d" * 40, False, (), "1" * 64),
        fake_sglang=SourceMetadata(
            "/fake-sglang", "f" * 40, True, (" M file",), "2" * 64
        ),
    )


class ManifestTests(unittest.TestCase):
    def _dgd(
        self,
        mode: str = "kv",
        enable_kv_return: bool = False,
        fake_output_token_id: int | None = None,
        frontend_replicas: int = 1,
        router_replica_sync: bool = False,
    ) -> dict[str, object]:
        return next(
            resource
            for resource in render_resources(
                "local/image:test",
                "local/planner:test",
                sources(),
                mode,
                enable_kv_return,
                fake_output_token_id,
                frontend_replicas,
                router_replica_sync,
            )
            if resource["kind"] == "DynamoGraphDeployment"
        )

    def test_kv_return_is_enabled_on_both_worker_tiers(self) -> None:
        components = self._dgd("kv", enable_kv_return=True)["spec"]["components"]
        for component in components:
            args = component["podTemplate"]["spec"]["containers"][0]["args"]
            self.assertEqual(
                "--enable-kv-return" in args,
                component["type"] in {"prefill", "decode"},
            )

    def test_fixed_output_token_is_enabled_on_both_worker_tiers(self) -> None:
        components = self._dgd(fake_output_token_id=264)["spec"]["components"]
        for component in components:
            args = component["podTemplate"]["spec"]["containers"][0]["args"]
            if component["type"] in {"prefill", "decode"}:
                self.assertEqual(
                    args[args.index("--fake-output-token-id") + 1],
                    "264",
                )
            else:
                self.assertNotIn("--fake-output-token-id", args)

    def test_validated_topology(self) -> None:
        components = self._dgd()["spec"]["components"]
        observed = {
            component["type"]: component["replicas"] for component in components
        }
        self.assertEqual(
            observed,
            {"frontend": 1, "planner": 1, "prefill": 4, "decode": 4},
        )

    def test_multiple_frontends_explicitly_disable_replica_sync(self) -> None:
        components = self._dgd(frontend_replicas=4)["spec"]["components"]
        frontend = next(
            component for component in components if component["type"] == "frontend"
        )
        self.assertEqual(frontend["replicas"], 4)
        args = frontend["podTemplate"]["spec"]["containers"][0]["args"]
        self.assertIn("--no-router-replica-sync", args)
        self.assertNotIn("--router-replica-sync", args)

    def test_router_replica_sync_can_be_enabled(self) -> None:
        components = self._dgd(
            frontend_replicas=4,
            router_replica_sync=True,
        )["spec"]["components"]
        frontend = next(
            component for component in components if component["type"] == "frontend"
        )
        args = frontend["podTemplate"]["spec"]["containers"][0]["args"]
        self.assertIn("--router-replica-sync", args)
        self.assertNotIn("--no-router-replica-sync", args)

    def test_workers_inherit_h200_defaults(self) -> None:
        components = self._dgd()["spec"]["components"]
        forbidden = {
            "--fake-num-kv-blocks",
            "--page-size",
            "--context-length",
            "--kv-cache-dtype",
            "--max-running-requests",
        }
        for component in components:
            if component["type"] not in {"prefill", "decode"}:
                continue
            args = component["podTemplate"]["spec"]["containers"][0]["args"]
            self.assertIn("--dp-size", args)
            self.assertEqual(args[args.index("--dp-size") + 1], "10")
            self.assertIn("--enable-dp-attention", args)
            self.assertFalse(forbidden.intersection(args))
        self.assertEqual(
            H200_PROFILE["kv_blocks_per_rank"]
            * H200_PROFILE["tokens_per_page"],
            H200_PROFILE["token_slots_per_rank"],
        )

    def test_frontend_uses_requested_routing_contract(self) -> None:
        expected = {
            "kv": ("kv", "--router-kv-events"),
            "load-aware": ("kv", "--load-aware"),
            "power-of-two": ("power-of-two", None),
        }
        for mode, (router_mode, mode_flag) in expected.items():
            components = self._dgd(mode)["spec"]["components"]
            for component in components:
                if component["type"] not in ("frontend", "prefill", "decode"):
                    continue
                args = component["podTemplate"]["spec"]["containers"][0]["args"]
                self.assertEqual(
                    args[args.index("--router-mode") + 1],
                    router_mode,
                    component["type"],
                )
                for flag in ("--router-kv-events", "--load-aware"):
                    self.assertEqual(
                        flag in args,
                        flag == mode_flag,
                        component["type"],
                    )

    def test_grafana_dashboard_configmap_is_rendered(self) -> None:
        configmap = render_grafana_dashboard()
        self.assertEqual(configmap["metadata"]["namespace"], "observability")
        dashboard = configmap["data"]["dynamo-lab.json"]
        self.assertIn('"uid":"dynamo-local-sglang"', dashboard)
        self.assertIn("simulated_prefix_cache_hit_tokens_total", dashboard)
        self.assertIn("dynamo_component_router_kv_hit_rate_sum", dashboard)

    def test_grafana_dashboard_matches_dynamo_sim(self) -> None:
        dashboard = (
            Path(__file__).resolve().parents[1] / "config" / "grafana-dashboard.json"
        )
        digest = hashlib.sha256(dashboard.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            "4094805e0583ec445ade5070152ba2b9fd5be758c4c6c9d327881715185d5a58",
        )

    def test_prometheus_aliases_dynamo_16_metrics_for_dashboard(self) -> None:
        config = (
            Path(__file__).resolve().parents[1] / "config" / "prometheus.yaml"
        ).read_text()
        for metric in (
            "dynamo_frontend_output_tokens_total",
            "dynamo_frontend_time_to_first_token_seconds_bucket",
            "dynamo_frontend_inter_token_latency_seconds_bucket",
            "dynamo_frontend_active_requests",
            "dynamo_frontend_queued_requests",
        ):
            self.assertIn(f"record: {metric}", config)
        self.assertNotIn("record: dynamo_frontend_requests_total", config)

    def test_otel_collector_bounds_churned_metric_memory(self) -> None:
        config = (
            Path(__file__).resolve().parents[1] / "config" / "otel-collector.yaml"
        ).read_text()
        self.assertIn("limit_mib: 1536", config)
        self.assertIn("metric_expiration: 5m", config)
        self.assertIn('limits: {cpu: "1", memory: 2Gi}', config)

    def test_tempo_bounds_wal_blocks_with_sufficient_memory(self) -> None:
        config = (
            Path(__file__).resolve().parents[1] / "config" / "tempo.yaml"
        ).read_text()
        self.assertIn("max_block_duration: 5m", config)
        self.assertIn("max_block_bytes: 100000000", config)
        self.assertIn("limits: {cpu: 500m, memory: 2Gi}", config)

    def test_source_commits_are_annotated(self) -> None:
        annotations = self._dgd()["metadata"]["annotations"]
        self.assertEqual(annotations["dynamo-sim.local/dynamo-commit"], "d" * 40)
        self.assertEqual(
            annotations["dynamo-sim.local/fake-sglang-commit"], "f" * 40
        )

    def test_planner_uses_local_source_image(self) -> None:
        components = self._dgd()["spec"]["components"]
        planner = next(item for item in components if item["type"] == "planner")
        image = planner["podTemplate"]["spec"]["containers"][0]["image"]
        self.assertEqual(image, "local/planner:test")


if __name__ == "__main__":
    unittest.main()
