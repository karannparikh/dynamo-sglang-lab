from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def load_contract_module():
    path = Path(__file__).resolve().parents[1] / "docker" / "validate_contract.py"
    spec = importlib.util.spec_from_file_location("validate_contract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ServerArgs:
    def __init__(
        self,
        model_path: str,
        fake_num_kv_blocks: int = 15_509,
        page_size: int = 256,
        context_length: int = 1_048_576,
        kv_cache_dtype: str = "fp8_e4m3",
        max_running_requests: int | None = None,
        dp_size: int = 1,
    ) -> None:
        self.model_path = model_path
        self.fake_num_kv_blocks = fake_num_kv_blocks
        self.page_size = page_size
        self.context_length = context_length
        self.kv_cache_dtype = kv_cache_dtype
        self.max_running_requests = max_running_requests
        self.dp_size = dp_size


class SimConfig:
    """Minimal stand-in mirroring fakeengine's per-rank derivation."""

    def __init__(self, max_running_requests: int) -> None:
        self.max_running_requests = max_running_requests

    @classmethod
    def from_server_args(cls, server_args: ServerArgs) -> "SimConfig":
        raw = server_args.max_running_requests
        dp_size = server_args.dp_size or 1
        return cls(raw // dp_size if raw is not None else 32)


class ContractTests(unittest.TestCase):
    def _modules(self) -> dict[str, types.ModuleType]:
        sglang = types.ModuleType("sglang")
        sglang.__is_fake_sglang__ = True
        srt = types.ModuleType("sglang.srt")
        server_args = types.ModuleType("sglang.srt.server_args")
        server_args.ServerArgs = ServerArgs
        fakeengine = types.ModuleType("fakeengine")
        fakeengine_config = types.ModuleType("fakeengine.config")
        fakeengine_config.SimConfig = SimConfig
        return {
            "sglang": sglang,
            "sglang.srt": srt,
            "sglang.srt.server_args": server_args,
            "fakeengine": fakeengine,
            "fakeengine.config": fakeengine_config,
        }

    def test_fake_sglang_effective_profile(self) -> None:
        contract = load_contract_module()
        with patch.dict(sys.modules, self._modules()):
            result = contract.validate()
        self.assertTrue(result["profile_inherited"])
        self.assertEqual(result["profile"]["token_slots_per_rank"], 3_970_304)
        self.assertEqual(
            result["profile"]["max_running_requests_per_rank"], 32
        )


if __name__ == "__main__":
    unittest.main()
