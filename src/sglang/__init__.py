"""A CPU-only stand-in for the ``sglang`` package.

This is NOT SGLang. It presents the subset of SGLang's Python API that
NVIDIA Dynamo's SGLang backend consumes, backed by :mod:`fakeengine`, which
simulates prefill, decode, and KV-cache transfer without a GPU or any model
weights. Importing this package where real SGLang is expected is intentional;
importing it by accident is not, hence the marker below and the startup banner
in :mod:`fakeengine.config`.
"""

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.server_args import ServerArgs

__version__ = "0.5.19"          # the SGLang release whose API this mirrors
__is_fake_sglang__ = True

__all__ = ["Engine", "ServerArgs", "__version__", "__is_fake_sglang__"]
