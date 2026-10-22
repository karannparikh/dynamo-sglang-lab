"""Process-wide resolved configuration (SGLang >= 0.5.19)."""

from typing import Any, Optional

_CONTEXT: dict[str, Any] = {}


def publish(server_args: Any, *, role: str = "engine") -> None:
    _CONTEXT[role] = server_args
    _CONTEXT["latest"] = server_args


def get_context(role: str = "latest") -> Optional[Any]:
    return _CONTEXT.get(role)


def reset_context() -> None:
    _CONTEXT.clear()
