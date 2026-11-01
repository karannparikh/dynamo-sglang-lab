from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


class CommandError(RuntimeError):
    pass


@dataclass(frozen=True)
class CommandResult:
    args: tuple[str, ...]
    stdout: str
    stderr: str
    returncode: int


class Runner:
    def run(
        self,
        args: Sequence[str | os.PathLike[str]],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        check: bool = True,
        timeout: float | None = None,
        input_text: str | None = None,
    ) -> CommandResult:
        argv = tuple(os.fspath(arg) for arg in args)
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        completed = subprocess.run(
            argv,
            cwd=cwd,
            env=merged_env,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        result = CommandResult(
            args=argv,
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
        )
        if check and completed.returncode:
            rendered = " ".join(argv)
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise CommandError(
                f"command failed ({completed.returncode}): {rendered}\n{detail}"
            )
        return result
