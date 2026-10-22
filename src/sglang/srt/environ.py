"""SGLang's env-var registry. Dynamo's snapshot path reads ``envs``."""

import os
from typing import Any


class _EnvField:
    def __init__(self, name: str, default: Any = None) -> None:
        self.name = name
        self.default = default

    def get(self) -> Any:
        return os.environ.get(self.name, self.default)

    def set(self, value: Any) -> None:
        os.environ[self.name] = str(value)

    def is_set(self) -> bool:
        return self.name in os.environ


class _Envs:
    """Attribute access mints a field on demand, matching SGLang's registry
    behavior closely enough that Dynamo's ``getattr(envs, name, None)`` probes
    resolve."""

    def __getattr__(self, name: str) -> _EnvField:
        field = _EnvField(name if name.startswith("SGLANG_") else f"SGLANG_{name.upper()}")
        setattr(self, name, field)
        return field


envs = _Envs()
