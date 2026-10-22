"""Speculative-decoding algorithm tags. Dynamo's ``register.py`` maps these
onto ``ModelRuntimeConfig.enable_eagle``."""

from enum import Enum
from typing import Optional


class SpeculativeAlgorithm(Enum):
    NONE = "NONE"
    EAGLE = "EAGLE"
    EAGLE3 = "EAGLE3"
    STANDALONE = "STANDALONE"
    NEXTN = "NEXTN"
    FROZEN_KV_MTP = "FROZEN_KV_MTP"

    def is_none(self) -> bool:
        return self is SpeculativeAlgorithm.NONE

    def is_eagle(self) -> bool:
        # Dynamo derives ModelRuntimeConfig.enable_eagle from this predicate so
        # the KV router bigram-aligns prompt block hashes the same way the
        # radix cache does. FROZEN_KV_MTP keys its KV events like eagle, so it
        # belongs here -- see register.py::_eagle_enabled_for.
        return self in (
            SpeculativeAlgorithm.EAGLE,
            SpeculativeAlgorithm.EAGLE3,
            SpeculativeAlgorithm.FROZEN_KV_MTP,
        )

    def is_eagle3(self) -> bool:
        return self is SpeculativeAlgorithm.EAGLE3

    def is_standalone(self) -> bool:
        return self is SpeculativeAlgorithm.STANDALONE

    @staticmethod
    def from_string(name: Optional[str]) -> "SpeculativeAlgorithm":
        """``None`` means "not speculative"; an unrecognized name is an error.

        Dynamo catches the ValueError and degrades gracefully; swallowing it
        here instead would silently disable eagle routing for a typo.
        """
        if not name:
            return SpeculativeAlgorithm.NONE
        try:
            return SpeculativeAlgorithm[name.upper()]
        except KeyError as exc:
            raise ValueError(f"unknown speculative algorithm: {name!r}") from exc
