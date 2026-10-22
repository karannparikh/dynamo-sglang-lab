"""Tracing hooks. The fake engine emits no spans of its own."""

import logging

_LEVEL = "off"


def set_global_trace_level(level) -> None:
    global _LEVEL
    _LEVEL = str(level)
    logging.getLogger(__name__).debug("fake-sglang trace level set to %s", _LEVEL)


def get_global_trace_level() -> str:
    return _LEVEL


def trace_set_remote_propagate_context(*args, **kwargs) -> None:
    return None
