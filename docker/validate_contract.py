from __future__ import annotations

import argparse
import importlib
import inspect
import json
from pathlib import Path
from typing import Any

EXPECTED = {
    "kv_blocks_per_rank": 15_509,
    "tokens_per_page": 256,
    "token_slots_per_rank": 3_970_304,
    "context_length": 1_048_576,
    "kv_cache_dtype": "fp8_e4m3",
    "max_running_requests_per_rank": 32,
}

ALIASES = {
    "kv_blocks_per_rank": ("fake_num_kv_blocks", "num_kv_blocks"),
    "tokens_per_page": ("page_size", "tokens_per_page"),
    "context_length": ("context_length", "max_model_len"),
    "kv_cache_dtype": ("kv_cache_dtype",),
}


def _server_args() -> Any:
    sglang = importlib.import_module("sglang")
    if not getattr(sglang, "__is_fake_sglang__", False):
        raise RuntimeError(
            "FAKE_SGLANG_SOURCE installed an sglang package without "
            "__is_fake_sglang__=True"
        )
    module = importlib.import_module("sglang.srt.server_args")
    server_args = getattr(module, "ServerArgs", None)
    if server_args is None:
        raise RuntimeError("sglang.srt.server_args.ServerArgs is missing")
    signature = inspect.signature(server_args)
    kwargs: dict[str, object] = {}
    for name, parameter in signature.parameters.items():
        if parameter.kind in (
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            continue
        if parameter.default is not inspect.Parameter.empty:
            continue
        if name in {"model_path", "model"}:
            kwargs[name] = "/opt/fake-sglang/models/qwen3-0.6b"
        elif name in {"served_model_name", "served_model"}:
            kwargs[name] = "fake-model"
        else:
            raise RuntimeError(
                f"cannot construct ServerArgs; unsupported required parameter: {name}"
            )
    return server_args(**kwargs)


def _read_value(instance: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        if hasattr(instance, name):
            value = getattr(instance, name)
            return getattr(value, "value", value)
    raise RuntimeError(f"effective fake-sglang profile is missing {names!r}")


def _effective_max_running_requests(server_args: Any) -> int:
    """Per-rank concurrent-decode cap as fake-sglang actually resolves it.

    ``ServerArgs.max_running_requests`` is a per-process total that defaults to
    ``None``; the engine derives the per-rank value in
    ``fakeengine.config.SimConfig.from_server_args`` (``None`` -> 32). Read that
    effective value instead of the raw attribute.
    """
    module = importlib.import_module("fakeengine.config")
    sim_config = getattr(module, "SimConfig")
    return int(sim_config.from_server_args(server_args).max_running_requests)


def validate() -> dict[str, object]:
    instance = _server_args()
    observed = {
        field: _read_value(instance, aliases) for field, aliases in ALIASES.items()
    }
    observed["token_slots_per_rank"] = (
        int(observed["kv_blocks_per_rank"]) * int(observed["tokens_per_page"])
    )
    observed["max_running_requests_per_rank"] = _effective_max_running_requests(
        instance
    )
    failures = {
        field: {"expected": expected, "observed": observed.get(field)}
        for field, expected in EXPECTED.items()
        if observed.get(field) != expected
    }
    if failures:
        raise RuntimeError(
            "fake-sglang effective H200 profile mismatch: "
            + json.dumps(failures, sort_keys=True)
        )
    return {
        "contract": "fake-sglang-h200-application-defaults-v1",
        "profile": observed,
        "profile_inherited": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
