"""Scheduler process entry point.

The fake engine runs its scheduler in-process as an asyncio task rather than
forking a rank per GPU, so this exists only as the override point Dynamo's
NIXL telemetry wrapper patches (``nixl_telemetry.py``). Reaching the body means
something asked for a real multi-process launch.
"""

from typing import Any


def run_scheduler_process(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(
        "fake-sglang runs its scheduler in-process; there is no scheduler "
        "subprocess to launch."
    )
