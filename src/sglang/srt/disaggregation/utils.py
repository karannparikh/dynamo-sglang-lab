"""Disaggregation constants and the KV-transfer state machine.

``FAKE_BOOTSTRAP_HOST`` and ``KVPoll`` are lifted verbatim from SGLang
(``python/sglang/srt/disaggregation/utils.py`` and ``base/conn.py``). Dynamo's
health-check canaries and ``_disagg.warmup_prefill_engine`` send requests with
``bootstrap_host == FAKE_BOOTSTRAP_HOST``; those must complete instantly and
locally rather than wait on a peer, or the warmup blocks for its 1800 s
timeout.
"""

from enum import Enum

# SGLang's sentinel for "do not actually transfer".
FAKE_BOOTSTRAP_HOST = "2.2.2.2"


class DisaggregationMode(Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"
    ENCODE = "encode"


class TransferBackend(Enum):
    MOONCAKE = "mooncake"
    NIXL = "nixl"
    ASCEND = "ascend"
    MORI = "mori"
    FAKE = "fake"


class KVPoll:
    """Plain ints, not an Enum: SGLang MIN-reduces these across ranks so that
    ``Failed`` dominates and no rank commits ahead of its peers."""

    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4

    _NAMES = {
        0: "Failed",
        1: "Bootstrapping",
        2: "WaitingForInput",
        3: "Transferring",
        4: "Success",
    }

    @classmethod
    def name(cls, value: int) -> str:
        return cls._NAMES.get(value, str(value))


def is_fake_bootstrap_host(host) -> bool:
    return host == FAKE_BOOTSTRAP_HOST
