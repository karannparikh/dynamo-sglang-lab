import asyncio

import pytest

from fakeengine.config import SimConfig
from fakeengine.stats import MetricsPusher
from sglang.srt.entrypoints.engine import Engine


async def generate(engine, rank, tokens, rid):
    stream = await engine.async_generate(
        input_ids=tokens, data_parallel_rank=rank, rid=rid,
        sampling_params={"max_new_tokens": 1},
    )
    return [chunk async for chunk in stream]


def test_packed_capacity_is_per_rank(server_args):
    config = SimConfig.from_server_args(server_args(
        dp_size=10, max_running_requests=80, fake_num_kv_blocks=8192, page_size=16,
    ))
    assert config.max_running_requests == 8
    assert config.total_kv_tokens == 131072


@pytest.mark.parametrize("options", [
    {"dp_size": 0}, {"dp_size": -1},
    {"dp_size": 2, "max_running_requests": 3},
    {"dp_size": 2, "enable_dp_attention": False},
    {"dp_size": 2, "enable_dp_attention": True, "nnodes": 2},
])
def test_invalid_packed_configuration_fails(server_args, options):
    with pytest.raises(ValueError):
        Engine(server_args=server_args(**options))


async def test_rank_cache_load_and_flush_are_isolated(server_args):
    engine = Engine(server_args=server_args(
        dp_size=2, enable_dp_attention=True, page_size=4, fake_num_kv_blocks=32,
        max_running_requests=16,
    ))
    try:
        prompt = list(range(16))
        cold0 = await generate(engine, 0, prompt, "cold0")
        cold1 = await generate(engine, 1, prompt, "cold1")
        warm0 = await generate(engine, 0, prompt, "warm0")
        assert cold0[-1]["meta_info"]["cached_tokens"] == 0
        assert cold1[-1]["meta_info"]["cached_tokens"] == 0
        assert warm0[-1]["meta_info"]["cached_tokens"] == 16
        assert engine.schedulers[0].pool is not engine.schedulers[1].pool
        assert [s.prefix_cache_queries for s in engine.schedulers] == [2, 1]
        assert len(engine._scheduler_init_result.scheduler_infos) == 2
        for rank, scheduler in enumerate(engine.schedulers):
            metrics = MetricsPusher(scheduler, "unused", dp_rank=rank)._snapshot()
            assert metrics.data_parallel_rank == rank
            assert metrics.kv_total_blocks == 128
            assert metrics.request_total_slots == 8
        await engine.tokenizer_manager.flush_cache()
        assert all(s.pool.cached_blocks == 0 for s in engine.schedulers)
    finally:
        await engine._async_shutdown()


async def test_one_rank_can_progress_while_another_is_busy(server_args):
    engine = Engine(server_args=server_args(dp_size=2, enable_dp_attention=True))
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked_sleep(_milliseconds):
        entered.set()
        await release.wait()

    engine.schedulers[0].timing.sleep_ms = blocked_sleep
    task = asyncio.create_task(generate(engine, 0, list(range(16)), "blocked"))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        result = await asyncio.wait_for(generate(engine, 1, [101, 102], "free"), 2)
        assert result[-1]["meta_info"]["completion_tokens"] == 1
        assert not task.done()
        engine.tokenizer_manager.abort_request("blocked")
        assert engine.schedulers[0].by_rid.get("blocked") is None
        assert engine.schedulers[1].prefix_cache_queries == 1
    finally:
        release.set()
        await task
        await engine._async_shutdown()


@pytest.mark.parametrize("rank", [-1, 2, True, "1"])
async def test_invalid_routed_rank_fails_without_admission(server_args, rank):
    engine = Engine(server_args=server_args(dp_size=2, enable_dp_attention=True))
    try:
        with pytest.raises(ValueError, match="Invalid routed DP rank"):
            await engine.async_generate(input_ids=[1, 2], data_parallel_rank=rank)
        assert all(not s.by_rid for s in engine.schedulers)
    finally:
        await engine._async_shutdown()
