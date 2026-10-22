"""CPU simulation behind the fake SGLang API.

Nothing here knows about SGLang's API shape -- :mod:`sglang` is the translation
layer. This package owns the behavior: a continuous-batching scheduler, a KV
block pool, a latency model, and a real cross-process bootstrap handshake with
a simulated KV-cache transfer.
"""

from fakeengine.config import SimConfig

__all__ = ["SimConfig"]
