"""``--config <file>`` merging.

Dynamo builds one of these around its SGLang-only parser when the launch passes
``--config``. Values from the file are inserted as CLI flags *behind* what the
command line already supplied, so an explicit flag always wins.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, List


class ConfigArgumentMerger:
    def __init__(self, parser: argparse.ArgumentParser) -> None:
        self.parser = parser

    def merge_config_with_args(self, args: List[str]) -> List[str]:
        path = self._pop_config_path(args)
        if path is None:
            return args
        merged = list(self._config_to_flags(self._load(path)))
        merged.extend(args)                # command line last => command line wins
        return merged

    @staticmethod
    def _pop_config_path(args: List[str]) -> Any:
        for index, token in enumerate(args):
            if token == "--config" and index + 1 < len(args):
                path = args[index + 1]
                del args[index : index + 2]
                return path
            if token.startswith("--config="):
                path = token.split("=", 1)[1]
                del args[index]
                return path
        return None

    @staticmethod
    def _load(path: str) -> dict:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"config file not found: {path}")
        with open(path) as handle:
            text = handle.read()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                import yaml
            except ImportError as exc:
                raise ValueError(
                    f"{path} is not JSON and PyYAML is not installed"
                ) from exc
            return yaml.safe_load(text) or {}

    @staticmethod
    def _config_to_flags(config: dict):
        for key, value in config.items():
            flag = "--" + str(key).replace("_", "-")
            if isinstance(value, bool):
                if value:
                    yield flag
            elif isinstance(value, (list, tuple)):
                for item in value:
                    yield flag
                    yield str(item)
            elif value is not None:
                yield flag
                yield str(value)
