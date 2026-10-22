import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "mooncake_lab", Path(__file__).resolve().parents[1] / "scripts/mooncake_lab.py",
)
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


def test_routing_comparison_preserves_admission_and_engine_cache():
    original = [
        "--router-mode", "kv", "--router-kv-events",
        "--router-policy-config", "/etc/lab-router/router-policy.yaml",
        "--engine-request-limit", "8", "--kv-events-config", '{"publisher":"zmq"}',
    ]
    off = lab.routing_args(original, "load-aware")
    assert "--router-kv-events" not in off
    assert off[-1] == "--load-aware"
    assert off[:-1] == [a for a in original if a != "--router-kv-events"]
    assert "--router-kv-events" in original
    on = lab.routing_args(off, "kv")
    assert "--load-aware" not in on
    assert on[-1] == "--router-kv-events"
    assert on[:-1] == off[:-1]
    assert lab.routing_args(original, "kv") == original


@pytest.mark.parametrize("previous_mode", ["kv", "load-aware", "power-of-two"])
@pytest.mark.parametrize("mode", ["kv", "load-aware", "power-of-two"])
def test_all_routing_mode_transitions(previous_mode, mode):
    original = [
        "--router-mode", "kv", "--router-kv-events",
        "--dp-size", "10", "--engine-request-limit", "80",
        "--kv-events-config", '{"publisher":"zmq"}',
    ]
    previous = lab.routing_args(original, previous_mode)
    result = lab.routing_args(previous, mode)
    expected_router_mode = "power-of-two" if mode == "power-of-two" else "kv"
    assert result[result.index("--router-mode") + 1] == expected_router_mode
    assert ("--router-kv-events" in result) == (mode == "kv")
    assert ("--load-aware" in result) == (mode == "load-aware")
    assert [arg for arg in result[2:] if arg not in ("--router-kv-events", "--load-aware")] == [
        "--dp-size", "10", "--engine-request-limit", "80",
        "--kv-events-config", '{"publisher":"zmq"}',
    ]
    assert lab.routing_args(result, mode) == result
    assert original[1:3] == ["kv", "--router-kv-events"]


def test_routing_args_adds_missing_mode_and_rejects_unknown_mode():
    assert lab.routing_args([], "power-of-two") == ["--router-mode", "power-of-two"]
    with pytest.raises(ValueError, match="Unknown routing mode"):
        lab.routing_args([], "invalid")


def test_kubectl_always_targets_the_dedicated_local_context():
    assert lab.KUBECTL[lab.KUBECTL.index("--context") + 1] == "kind-dynamo-sim"
    assert lab.KUBECTL[lab.KUBECTL.index("--kubeconfig") + 1] == str(lab.STATE / "kubeconfig")


def test_cpu_interpolation_uses_sample_timestamps_not_poll_frequency():
    report_spec = importlib.util.spec_from_file_location(
        "mooncake_report", Path(__file__).resolve().parents[1] / "scripts/mooncake_report.py",
    )
    report = importlib.util.module_from_spec(report_spec)
    report_spec.loader.exec_module(report)
    cpu, covered = report.integrate({0: 20, 10: 25, 20: 40}, 5, 15)
    assert cpu == 10
    assert covered == 10
    assert report.integrate({0: 20, 10: 25}, 20, 30) == (0, 0)
    result = report.cpu_report({
        ("prefill", "pod1", "main", "start"): {0: 0, 10: 10},
        ("prefill", "pod2", "main", "start"): {0: 0, 10: 10},
    }, 0, 10, {"prefill": 2})
    assert result["prefill"]["mean_cores"] == 2
    assert result["prefill"]["coverage_fraction"] == 1
