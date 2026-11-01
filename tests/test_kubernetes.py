from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from dynamo_routing_harness.kubernetes import (
    FleetValidationError,
    Kubernetes,
    parse_fleet,
    validate_fleet,
)


def pod(name: str, role: str, index: int, *, ready: bool = True):
    args = []
    if role in {"prefill", "decode"}:
        args = [
            "--dp-size",
            "10",
            "--enable-dp-attention",
            "--model-path",
            "/opt/fake-sglang/models/qwen3-0.6b",
            "--served-model-name",
            "fake-model",
        ]
    return {
        "metadata": {
            "name": name,
            "uid": f"uid-{index}",
            "labels": {"nvidia.com/dynamo-component-type": role},
        },
        "spec": {
            "nodeName": "worker",
            "containers": [{"name": "main", "args": args}],
        },
        "status": {
            "phase": "Running",
            "conditions": [{"type": "Ready", "status": "True" if ready else "False"}],
            "containerStatuses": [{"name": "main", "restartCount": 0}],
        },
    }


def complete_payload():
    items = [pod("frontend-0", "frontend", 0), pod("planner-0", "planner", 1)]
    items.extend(pod(f"prefill-{i}", "prefill", 10 + i) for i in range(4))
    items.extend(pod(f"decode-{i}", "decode", 20 + i) for i in range(4))
    return {"items": items}


class KubernetesTests(unittest.TestCase):
    def test_observability_namespace_is_applied_before_resources(self) -> None:
        class RecordingRunner:
            def __init__(self) -> None:
                self.calls = []

            def run(self, args, **kwargs):
                self.calls.append((tuple(args), kwargs))
                return SimpleNamespace(returncode=0, stdout="", stderr="")

        runner = RecordingRunner()
        Kubernetes(runner=runner).install_observability()
        args, kwargs = runner.calls[0]
        self.assertEqual(args, ("kubectl", "apply", "-f", "-"))
        namespace = json.loads(kwargs["input_text"])
        self.assertEqual(namespace["metadata"]["name"], "observability")

    def test_wait_for_nodes_retries_api_bootstrap_failure(self) -> None:
        class SequenceRunner:
            def __init__(self) -> None:
                self.results = iter(
                    (
                        SimpleNamespace(
                            returncode=1,
                            stdout="",
                            stderr='nodes is forbidden: User "kubernetes-admin"',
                        ),
                        SimpleNamespace(
                            returncode=0,
                            stdout='{"items":[{"metadata":{"name":"worker"}}]}',
                            stderr="",
                        ),
                    )
                )
                self.calls = 0

            def run(self, args, **kwargs):
                self.calls += 1
                return next(self.results)

        runner = SequenceRunner()
        nodes = Kubernetes(runner=runner)._wait_for_nodes(
            timeout=1,
            interval=0,
        )
        self.assertEqual(nodes["items"][0]["metadata"]["name"], "worker")
        self.assertEqual(runner.calls, 2)

    def test_complete_fleet_validates(self) -> None:
        snapshot = parse_fleet(complete_payload())
        validate_fleet(snapshot)
        self.assertEqual(len(snapshot.pods), 10)
        self.assertEqual(len(snapshot.by_role("prefill")), 4)

    def test_not_ready_pod_fails(self) -> None:
        payload = complete_payload()
        payload["items"][2]["status"]["conditions"][0]["status"] = "False"
        with self.assertRaisesRegex(FleetValidationError, "ready=False"):
            validate_fleet(parse_fleet(payload))

    def test_profile_override_fails(self) -> None:
        payload = complete_payload()
        payload["items"][2]["spec"]["containers"][0]["args"].extend(
            ["--page-size", "256"]
        )
        with self.assertRaisesRegex(FleetValidationError, "overridden"):
            validate_fleet(parse_fleet(payload))

    def test_stability_includes_uid_and_restarts(self) -> None:
        baseline = parse_fleet(complete_payload())
        changed = complete_payload()
        changed["items"][0]["metadata"]["uid"] = "replacement"
        self.assertFalse(parse_fleet(changed).stable_against(baseline))


if __name__ == "__main__":
    unittest.main()
