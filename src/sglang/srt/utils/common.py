"""Grab-bag SGLang helpers Dynamo probes for.

The device predicates are ``lru_cache``d exactly as in real SGLang -- Dynamo's
tests call ``is_cpu.cache_clear()`` between monkeypatched environments, so the
decorator is part of the contract, not an optimization.
"""

import os
from functools import lru_cache


@lru_cache(maxsize=None)
def is_cpu() -> bool:
    return True


@lru_cache(maxsize=None)
def is_cuda() -> bool:
    return False


@lru_cache(maxsize=None)
def is_hip() -> bool:
    return False


@lru_cache(maxsize=None)
def get_device() -> str:
    return "cpu"


def get_bool_env_var(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


def load_video(*args, **kwargs):
    raise NotImplementedError("fake-sglang does not decode video")
