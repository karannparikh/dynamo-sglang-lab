"""Late-resolution overrides (SGLang >= 0.5.18).

Real SGLang keeps ``ServerArgs`` raw and resolves an effective projection
separately. The fake engine has no separate resolution pass, so a declared
override is applied directly and recorded for debugging.
"""

from typing import Any


def declare_late_resolution(server_args: Any, source: str, **fields: Any) -> None:
    declared = getattr(server_args, "_late_resolution", None)
    if declared is None:
        declared = {}
        object.__setattr__(server_args, "_late_resolution", declared)
    declared[source] = dict(fields)
    for name, value in fields.items():
        setattr(server_args, name, value)


def model_config_of(server_args: Any) -> Any:
    return server_args.get_model_config()


def use_mla_backend(server_args: Any) -> bool:
    return bool(server_args.use_mla_backend())
