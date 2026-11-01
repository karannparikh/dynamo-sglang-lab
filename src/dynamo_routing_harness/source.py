from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from .command import CommandError, Runner
from .constants import (
    DYNAMO_VANILLA_REF,
    DYNAMO_VANILLA_REPO,
    FAKE_SGLANG_VANILLA_REF,
    FAKE_SGLANG_VANILLA_REPO,
    LOCAL_ROOT,
)


class SourceValidationError(ValueError):
    pass


@dataclass(frozen=True)
class SourceMetadata:
    path: str
    head: str
    dirty: bool
    status: tuple[str, ...]
    fingerprint: str
    requested: str | None = None
    ref: str | None = None
    remote_url: str | None = None
    prebuilt: bool = False

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["status"] = list(self.status)
        return value


@dataclass(frozen=True)
class SourcePair:
    dynamo: SourceMetadata
    fake_sglang: SourceMetadata

    def as_dict(self) -> dict[str, object]:
        return {
            "dynamo": self.dynamo.as_dict(),
            "fake_sglang": self.fake_sglang.as_dict(),
        }


@dataclass(frozen=True)
class SourceSpec:
    name: str
    requested: str
    ref: str | None
    local_path: Path | None = None
    remote_url: str | None = None
    prebuilt: bool = False

    @property
    def is_remote(self) -> bool:
        return self.remote_url is not None


_GITHUB_HTTPS = re.compile(
    r"^(?:https://)?github\.com/(?P<owner>[A-Za-z0-9_.-]+)/"
    r"(?P<repo>[A-Za-z0-9_.-]+?)(?:\.git)?/?$"
)
_GITHUB_SSH = re.compile(
    r"^git@github\.com:(?P<owner>[A-Za-z0-9_.-]+)/"
    r"(?P<repo>[A-Za-z0-9_.-]+?)(?:\.git)?$"
)
_GITHUB_SHORTHAND = re.compile(
    r"^(?:github:)?(?P<owner>[A-Za-z0-9_.-]+)/"
    r"(?P<repo>[A-Za-z0-9_.-]+)$"
)


def _github_url(value: str) -> str | None:
    ssh_match = _GITHUB_SSH.fullmatch(value)
    if ssh_match:
        owner = ssh_match.group("owner")
        repo = ssh_match.group("repo")
        return f"git@github.com:{owner}/{repo}.git"
    for pattern in (_GITHUB_HTTPS, _GITHUB_SHORTHAND):
        match = pattern.fullmatch(value)
        if match:
            owner = match.group("owner")
            repo = match.group("repo")
            return f"https://github.com/{owner}/{repo}.git"
    return None


def parse_source_spec(
    name: str,
    value: str,
    ref: str | None = None,
) -> SourceSpec:
    candidate = Path(value).expanduser()
    explicit_local = (
        candidate.exists()
        or candidate.is_absolute()
        or value.startswith((".", "~"))
    )
    remote_url = None if explicit_local else _github_url(value)
    if remote_url is not None:
        return SourceSpec(
            name=name,
            requested=value,
            ref=ref or "HEAD",
            remote_url=remote_url,
        )
    if "://" in value or value.startswith("git@"):
        raise SourceValidationError(
            f"{name} remote sources must be GitHub locations, got: {value}"
        )
    if ref:
        raise SourceValidationError(
            f"{name}_REF applies only to remote GitHub sources; "
            f"local checkout {value} is never mutated"
        )
    return SourceSpec(
        name=name,
        requested=value,
        ref=None,
        local_path=candidate.resolve(),
    )


def selected_source_specs(
    environ: Mapping[str, str] | None = None,
) -> tuple[SourceSpec, SourceSpec]:
    values = os.environ if environ is None else environ
    dynamo_source = values.get("DYNAMO_SOURCE")
    if dynamo_source is None:
        dynamo = SourceSpec(
            name="DYNAMO",
            requested=f"{DYNAMO_VANILLA_REPO}@{DYNAMO_VANILLA_REF}",
            ref=DYNAMO_VANILLA_REF,
            remote_url=f"https://github.com/{DYNAMO_VANILLA_REPO}.git",
            prebuilt=True,
        )
    else:
        dynamo = parse_source_spec(
            "DYNAMO",
            dynamo_source,
            values.get("DYNAMO_REF"),
        )
    fake_source = values.get("FAKE_SGLANG_SOURCE")
    if fake_source is None:
        fake_sglang = SourceSpec(
            name="FAKE_SGLANG",
            requested=f"{FAKE_SGLANG_VANILLA_REPO}@{FAKE_SGLANG_VANILLA_REF}",
            ref=FAKE_SGLANG_VANILLA_REF,
            remote_url=f"https://github.com/{FAKE_SGLANG_VANILLA_REPO}.git",
        )
    else:
        fake_sglang = parse_source_spec(
            "FAKE_SGLANG",
            fake_source,
            values.get("FAKE_SGLANG_REF"),
        )
    return dynamo, fake_sglang


def _git(runner: Runner, path: Path, *args: str) -> str:
    try:
        return runner.run(("git", "-C", path, *args)).stdout.strip()
    except CommandError as error:
        raise SourceValidationError(
            f"{path} is not a usable git checkout: {error}"
        ) from error


def _fingerprint(runner: Runner, path: Path, head: str, status: str) -> str:
    digest = hashlib.sha256()
    digest.update(head.encode())
    digest.update(b"\0")
    digest.update(status.encode())
    digest.update(b"\0")
    if status:
        diff = _git(runner, path, "diff", "--binary", "HEAD")
        digest.update(diff.encode())
        untracked = _git(
            runner,
            path,
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        )
        for relative in sorted(filter(None, untracked.split("\0"))):
            digest.update(relative.encode())
            digest.update(b"\0")
            file_path = path / relative
            if file_path.is_file():
                digest.update(file_path.read_bytes())
    return digest.hexdigest()


def materialize_source(
    spec: SourceSpec,
    runner: Runner | None = None,
) -> Path:
    if spec.local_path is not None:
        return spec.local_path
    if spec.remote_url is None or spec.ref is None:
        raise SourceValidationError(f"invalid source specification: {spec!r}")
    active_runner = runner or Runner()
    key = hashlib.sha256(spec.remote_url.encode()).hexdigest()[:12]
    target = LOCAL_ROOT / "sources" / f"{spec.name.lower()}-{key}"
    freshly_cloned = not target.exists()
    if freshly_cloned:
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            active_runner.run(
                (
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    spec.remote_url,
                    target,
                )
            )
        except CommandError as error:
            raise SourceValidationError(
                f"failed to clone {spec.name} from {spec.remote_url}: {error}"
            ) from error
    if not (target / ".git").is_dir():
        raise SourceValidationError(
            f"remote source cache is not a Git checkout: {target}"
        )
    origin = _git(active_runner, target, "remote", "get-url", "origin")
    if origin.rstrip("/") != spec.remote_url.rstrip("/"):
        raise SourceValidationError(
            f"remote source cache {target} has unexpected origin {origin!r}; "
            f"expected {spec.remote_url!r}"
        )
    # A fresh `--no-checkout` clone has an empty working tree, so skip the guard
    # that protects an already-materialized cache from being clobbered.
    if not freshly_cloned:
        status = _git(
            active_runner,
            target,
            "status",
            "--porcelain=v1",
            "--untracked-files=normal",
        )
        if status:
            raise SourceValidationError(
                f"remote source cache is dirty and will not be overwritten: {target}"
            )
    try:
        active_runner.run(
            ("git", "-C", target, "fetch", "--depth=1", "origin", spec.ref)
        )
        active_runner.run(
            ("git", "-C", target, "checkout", "--detach", "FETCH_HEAD")
        )
    except CommandError as error:
        raise SourceValidationError(
            f"failed to materialize {spec.name} ref {spec.ref!r} "
            f"from {spec.remote_url}: {error}"
        ) from error
    return target.resolve()


def inspect_source(
    path: Path,
    runner: Runner | None = None,
    *,
    requested: str | None = None,
    ref: str | None = None,
    remote_url: str | None = None,
    prebuilt: bool = False,
) -> SourceMetadata:
    active_runner = runner or Runner()
    if not path.is_dir():
        raise SourceValidationError(f"source checkout does not exist: {path}")
    inside = _git(active_runner, path, "rev-parse", "--is-inside-work-tree")
    if inside != "true":
        raise SourceValidationError(f"source path is not a git work tree: {path}")
    head = _git(active_runner, path, "rev-parse", "HEAD")
    status_text = _git(
        active_runner,
        path,
        "status",
        "--porcelain=v1",
        "--untracked-files=normal",
    )
    status = tuple(line for line in status_text.splitlines() if line)
    return SourceMetadata(
        path=os.fspath(path),
        head=head,
        dirty=bool(status),
        status=status,
        fingerprint=_fingerprint(active_runner, path, head, status_text),
        requested=requested,
        ref=ref,
        remote_url=remote_url,
        prebuilt=prebuilt,
    )


def inspect_sources(
    runner: Runner | None = None,
    environ: Mapping[str, str] | None = None,
) -> SourcePair:
    active_runner = runner or Runner()
    dynamo_spec, fake_sglang_spec = selected_source_specs(environ)
    dynamo_path = materialize_source(dynamo_spec, active_runner)
    fake_sglang_path = materialize_source(fake_sglang_spec, active_runner)
    return SourcePair(
        dynamo=inspect_source(
            dynamo_path,
            active_runner,
            requested=dynamo_spec.requested,
            ref=dynamo_spec.ref,
            remote_url=dynamo_spec.remote_url,
            prebuilt=dynamo_spec.prebuilt,
        ),
        fake_sglang=inspect_source(
            fake_sglang_path,
            active_runner,
            requested=fake_sglang_spec.requested,
            ref=fake_sglang_spec.ref,
            remote_url=fake_sglang_spec.remote_url,
            prebuilt=fake_sglang_spec.prebuilt,
        ),
    )


def write_source_record(sources: SourcePair, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(sources.as_dict(), indent=2, sort_keys=True) + "\n")
