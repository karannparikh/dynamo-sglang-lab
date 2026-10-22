import json

import pytest

from deploy.local import render
from deploy.local.render import load_lab_config
from fakeengine.config import SimConfig
from fakeengine.kvcache import BlockPool, block_hashes, external_id
from fakeengine.scheduler import Request, Scheduler
from fakeengine.timing import TimingModel


@pytest.mark.parametrize("field,value", [
    ("fake_num_kv_blocks", 0), ("fake_num_kv_blocks", -1),
    ("page_size", 0), ("dcp_size", -1),
    ("max_total_tokens", 0), ("max_total_tokens", 17),
    ("fake_kv_generation_ms_per_1k_tokens", -1),
    ("fake_kv_generation_ms_per_1k_tokens", float("nan")),
    ("fake_kv_generation_ms_per_1k_tokens", float("inf")),
])
def test_invalid_cache_and_delay_settings_fail(server_args, field, value):
    with pytest.raises(ValueError):
        SimConfig.from_server_args(server_args(page_size=16, **{field: value})
                                   if field != "page_size" else server_args(page_size=value))


def test_token_capacity_overrides_block_count_and_is_advertised(server_args):
    from sglang.srt.entrypoints.engine import _SchedulerInitResult

    config = SimConfig.from_server_args(server_args(
        fake_num_kv_blocks=500, max_total_tokens=128, page_size=8, dcp_size=2,
    ))
    assert config.num_kv_blocks == 8
    assert config.block_size == 16
    assert config.total_kv_tokens == 128
    assert _SchedulerInitResult(config).scheduler_infos[0]["max_total_num_tokens"] == 128


def test_oversized_prompt_is_rejected_without_queueing(server_args):
    scheduler = Scheduler(SimConfig.from_server_args(server_args(
        fake_num_kv_blocks=8, page_size=4,
    )))
    with pytest.raises(ValueError, match="worker capacity is 32"):
        scheduler.submit(Request("too-large", list(range(33)), 1))
    assert scheduler.waiting == []
    assert scheduler.by_rid == {}


async def test_kv_generation_delay_only_bills_uncached_tokens(server_args):
    config = SimConfig.from_server_args(server_args(
        page_size=4, max_prefill_tokens=4, fake_kv_generation_ms_per_1k_tokens=100,
        fake_speedup_ratio=10,
    ))
    timing = TimingModel(config)
    assert timing.kv_generation_ms(1000) == 100
    assert timing.prefill_ms(1000) == config.prefill_overhead_ms + config.prefill_ms_per_1k + 100
    scheduler = Scheduler(config)
    try:
        for rid, prompt, prebuilt, enabled, expected_seconds in [
            ("cold", list(range(8)), False, True, 0.00008),
            ("warm", list(range(8)), False, True, 0.00008),
            ("extend", list(range(10)), False, True, 0.0001),
            ("transfer", list(range(12)), True, True, 0.0001),
            ("probe", list(range(100, 108)), False, False, 0.0001),
        ]:
            request = scheduler.submit(Request(
                rid, prompt, 1, prebuilt=prebuilt, observe_cache=enabled,
            ))
            while not request.finished:
                await scheduler._prefill_step()
            assert scheduler.kv_generation_seconds == pytest.approx(expected_seconds)
    finally:
        await scheduler.stop()


async def test_exact_capacity_prompt_finishes_cold_and_fully_cached(server_args):
    scheduler = Scheduler(SimConfig.from_server_args(server_args(
        fake_num_kv_blocks=2, page_size=4, max_prefill_tokens=4,
    )))
    try:
        for rid in ("cold", "warm"):
            request = scheduler.submit(Request(rid, list(range(8)), 1))
            for _ in range(2):
                assert await scheduler._prefill_step()
            assert request.finished
            assert request.cached_tokens == (8 if rid == "warm" else 0)
            assert scheduler.pool.used_blocks == 2
    finally:
        await scheduler.stop()


def test_eviction_keeps_published_prefix_parents_until_children_leave():
    removed = []
    pool = BlockPool(4, 4, on_evict=removed.extend)
    hashes = block_hashes(list(range(16)), 4)
    blocks = pool.allocate(4)
    pool.insert_prefix(hashes, blocks)
    pool.free(blocks)
    allocated = pool.allocate(1)
    assert removed == [external_id(hashes[-1])]
    assert pool.match_prefix(hashes) == 3
    pool.free(allocated)
    pool.clear_cache()
    assert removed == [external_id(h) for h in reversed(hashes)]
    assert pool.cached_blocks == 0
    assert pool.used_blocks == 0


def test_eviction_preserves_shared_parent_of_other_branches():
    pool = BlockPool(4, 2)
    left = block_hashes([1, 2, 3, 4], 2)
    blocks = pool.allocate(2)
    pool.insert_prefix(left, blocks)
    pool.free(blocks)
    right = block_hashes([1, 2, 5, 6], 2)
    shared = pool.acquire_prefix(right)
    blocks = shared + pool.allocate(1)
    pool.insert_prefix(right, blocks)
    pool.free(blocks)
    pool.allocate(2)
    assert pool.match_prefix(right) == 2
    assert pool.match_prefix(left) == 1
    assert pool.used_blocks <= pool.num_blocks


@pytest.mark.parametrize("publish", [False, True])
def test_pool_observes_pressure_eviction_separately_from_flush_and_protects_active_blocks(publish):
    observations, removed = [], []
    pool = BlockPool(
        3, 4, on_evict=removed.extend if publish else None,
        on_change=lambda: observations.append(
            (pool.cached_blocks, pool.used_blocks, pool.evictable_blocks,
             pool.evicted_blocks, pool.cleared_blocks),
        ),
    )
    hashes = block_hashes(list(range(12)), 4)
    blocks = pool.allocate(3)
    pool.insert_prefix(hashes, blocks)
    assert observations[-1] == (3, 3, 0, 0, 0)
    assert pool.allocate(1) is None
    pool.clear_cache()
    assert observations[-1] == (3, 3, 0, 0, 0)
    pool.free(blocks)
    assert observations[-1] == (3, 3, 3, 0, 0)
    shared = pool.acquire_prefix(hashes[:1])
    assert observations[-1] == (3, 3, 2, 0, 0)
    allocated = pool.allocate(1)
    assert observations[-1] == (2, 3, 1, 1, 0)
    pool.free(allocated)
    pool.clear_cache()
    assert observations[-1] == (1, 1, 0, 1, 1)
    pool.free(shared)
    pool.clear_cache()
    assert observations[-1] == (0, 0, 0, 1, 2)
    assert len(removed) == (3 if publish else 0)


@pytest.mark.parametrize("overrides", [
    {"kv_blocks_per_worker": 0}, {"kv_block_size": -1},
    {"kv_block_size": True}, {"publish_kv_events": "false"},
    {"context_length": 0}, {"context_length": True},
    {"ranks_per_process": 0}, {"worker_processes_per_role": True},
    {"kv_generation_ms_per_1k_tokens": float("nan")}, {"unexpected": 1},
])
def test_lab_configuration_rejects_invalid_settings(tmp_path, overrides):
    config = {
        "kv_blocks_per_worker": 128, "kv_block_size": 16, "context_length": 131072,
        "publish_kv_events": True, "kv_generation_ms_per_1k_tokens": 25.0,
        "worker_processes_per_role": 2, "ranks_per_process": 2,
        **overrides,
    }
    path = tmp_path / "lab.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_lab_config(path)


@pytest.mark.parametrize("publish", [True, False])
def test_lab_renderer_wires_capacity_delay_and_event_mode(monkeypatch, publish):
    monkeypatch.setattr(render, "load_lab_config", lambda: {
        "kv_blocks_per_worker": 128, "kv_block_size": 8, "context_length": 131072,
        "publish_kv_events": publish, "kv_generation_ms_per_1k_tokens": 250,
        "worker_processes_per_role": 2, "ranks_per_process": 2,
    })
    components = render.dynamo()[0]["spec"]["components"]
    for component in components:
        if component["type"] not in ("prefill", "decode", "frontend"):
            continue
        args = component["podTemplate"]["spec"]["containers"][0]["args"]
        assert "--router-kv-events" in args
        if component["type"] != "frontend":
            assert args[args.index("--context-length") + 1] == "131072"
            assert args[args.index("--page-size") + 1] == "8"
            assert args[args.index("--fake-num-kv-blocks") + 1] == "128"
            assert args[args.index("--fake-kv-generation-ms-per-1k-tokens") + 1] == "250"
            assert ("--kv-events-config" in args) == publish


def test_lab_renderer_pins_configured_processes_and_ranks():
    config = load_lab_config()
    components = {c["name"]: c for c in render.dynamo()[0]["spec"]["components"]}
    assert components["prefill"]["replicas"] == config["worker_processes_per_role"]
    assert components["decode"]["replicas"] == config["worker_processes_per_role"]
    args = components["Planner"]["podTemplate"]["spec"]["containers"][0]["args"]
    planner = json.loads(args[args.index("--config") + 1])
    assert planner["min_endpoint"] == config["worker_processes_per_role"]
    assert planner["enable_load_scaling"] is True
    minimum_gpu_cost = sum(
        planner["min_endpoint"]
        * int(components[role]["podTemplate"]["spec"]["containers"][0]
              ["resources"]["limits"]["nvidia.com/gpu"])
        for role in ("prefill", "decode")
    )
    assert planner["max_gpu_budget"] == minimum_gpu_cost
    launcher = (render.HERE / "lab.sh").read_text()
    assert '"nvidia.com/gpu":"10"' in launcher
    assert 'get("nvidia.com/gpu") == "10"' in launcher


def test_lab_capacity_fits_full_mooncake_sequences():
    config = load_lab_config()
    assert config["context_length"] == 131072
    assert config["kv_blocks_per_worker"] * config["kv_block_size"] == 131072


def test_context_override_reaches_simulator(server_args):
    config = SimConfig.from_server_args(server_args(
        context_length=131072, fake_num_kv_blocks=8192, page_size=16,
    ))
    assert config.context_len == 131072
    assert config.total_kv_tokens == 131072


def test_lab_admission_controls_are_bounded_and_mounted():
    ranks = load_lab_config()["ranks_per_process"]
    resources = render.dynamo()
    policy = next(r for r in resources if r["kind"] == "ConfigMap")
    assert f"request_queue_limit_per_worker: {10 * ranks}" in policy["data"]["router-policy.yaml"]
    assert f"raw_isl_token_queue_limit_per_worker: {65536 * ranks}" in policy["data"]["router-policy.yaml"]
    for component in resources[0]["spec"]["components"]:
        if component["type"] == "planner":
            continue
        pod = component["podTemplate"]["spec"]
        container = pod["containers"][0]
        assert "--router-policy-config" in container["args"]
        assert pod["volumes"][0]["configMap"]["name"] == policy["metadata"]["name"]
        if component["type"] != "frontend":
            args = container["args"]
            assert args[args.index("--engine-request-limit") + 1] == str(8 * ranks)
            assert args[args.index("--max-running-requests") + 1] == str(8 * ranks)
            assert args[args.index("--dp-size") + 1] == str(ranks)
            assert ("--enable-dp-attention" in args) is (ranks > 1)
            assert args[args.index("--max-prefill-tokens") + 1] == "8192"
            assert {"name": "DYN_DYNAMO_REQUEST_QUEUE_LIMIT", "value": str(2 * ranks)} in container["env"]
