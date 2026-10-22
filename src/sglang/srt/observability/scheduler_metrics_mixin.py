"""Scheduler metrics wire object.

Dynamo's ``publisher.py`` binds a ZMQ PULL socket on
``engine.port_args.metrics_ipc_name`` and expects the scheduler to push one of
these per interval; it reads ``data_parallel_rank``, ``kv_active_blocks``,
``kv_total_blocks`` (both in tokens -- Dynamo divides by ``page_size``) and
``gpu_cache_usage_perc``. Those feed load-based routing and the worker's
Prometheus gauges.
"""

from typing import Optional


class KvMetrics:
    __slots__ = (
        "request_active_slots", "request_total_slots",
        "kv_active_blocks", "kv_total_blocks",
        "num_requests_waiting", "gpu_cache_usage_perc", "gpu_prefix_cache_hit_rate",
        "data_parallel_rank",
    )

    def __init__(
        self,
        request_active_slots: int = 0,
        request_total_slots: int = 0,
        kv_active_blocks: int = 0,
        kv_total_blocks: int = 0,
        num_requests_waiting: int = 0,
        gpu_cache_usage_perc: float = 0.0,
        gpu_prefix_cache_hit_rate: float = 0.0,
        data_parallel_rank: Optional[int] = None,
    ) -> None:
        self.request_active_slots = request_active_slots
        self.request_total_slots = request_total_slots
        self.kv_active_blocks = kv_active_blocks
        self.kv_total_blocks = kv_total_blocks
        self.num_requests_waiting = num_requests_waiting
        self.gpu_cache_usage_perc = gpu_cache_usage_perc
        self.gpu_prefix_cache_hit_rate = gpu_prefix_cache_hit_rate
        self.data_parallel_rank = data_parallel_rank

    def __repr__(self) -> str:
        return (
            f"KvMetrics(running={self.request_active_slots}, "
            f"waiting={self.num_requests_waiting}, "
            f"kv={self.kv_active_blocks}/{self.kv_total_blocks})"
        )
