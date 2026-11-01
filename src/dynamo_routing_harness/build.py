from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .command import Runner
from .constants import (
    CLUSTER_NAME,
    DYNAMO_OPERATOR_IMAGE,
    DYNAMO_PLANNER_IMAGE,
    DYNAMO_RUNTIME_VERSION,
    DYNAMO_VERSION,
    LOCAL_ROOT,
    REPOSITORY_ROOT,
)
from .source import SourcePair


@dataclass(frozen=True)
class Images:
    serving: str
    planner: str
    operator: str

    @property
    def operator_repository(self) -> str:
        return self.operator.rsplit(":", 1)[0]

    @property
    def operator_tag(self) -> str:
        return self.operator.rsplit(":", 1)[1]


def image_names(sources: SourcePair) -> Images:
    fake_key = sources.fake_sglang.fingerprint[:12]
    if sources.dynamo.prebuilt:
        return Images(
            serving=f"local/dynamo-routing-serving:{fake_key}-{DYNAMO_RUNTIME_VERSION}",
            planner=DYNAMO_PLANNER_IMAGE,
            operator=DYNAMO_OPERATOR_IMAGE,
        )
    serving_key = f"{sources.dynamo.fingerprint[:12]}-{fake_key}"
    return Images(
        serving=f"local/dynamo-routing-serving:{serving_key}",
        planner=DYNAMO_PLANNER_IMAGE,
        operator=DYNAMO_OPERATOR_IMAGE,
    )


def colima_docker_env(home: Path | None = None) -> dict[str, str]:
    base = home or Path.home()
    return {
        "DOCKER_HOST": f"unix://{base}/.colima/dynamo-routing/docker.sock",
    }


class ImageBuilder:
    def __init__(self, runner: Runner | None = None) -> None:
        self.runner = runner or Runner()

    def build(self, sources: SourcePair) -> Images:
        images = image_names(sources)
        docker_env = colima_docker_env()
        if not self._image_exists(images.serving, docker_env):
            if sources.dynamo.prebuilt:
                self._build_serving_prebuilt(sources, images, docker_env)
            else:
                self._build_serving_source(sources, images, docker_env)
        # Only the serving image is built locally. The matching release's
        # operator and Planner images are pulled by the nodes from NGC.
        for image in (images.serving, images.planner, images.operator):
            if image.startswith("local/"):
                self.runner.run(
                    ("kind", "load", "docker-image", "--name", CLUSTER_NAME, image),
                    env=docker_env,
                )
        return images

    def _image_exists(
        self,
        image: str,
        docker_env: dict[str, str],
    ) -> bool:
        return (
            self.runner.run(
                ("docker", "image", "inspect", image),
                env=docker_env,
                check=False,
            ).returncode
            == 0
        )

    def _build_serving_prebuilt(
        self,
        sources: SourcePair,
        images: Images,
        docker_env: dict[str, str],
    ) -> None:
        self.runner.run(
            (
                "docker",
                "buildx",
                "build",
                "--platform",
                "linux/arm64",
                "--load",
                "--build-context",
                f"fake_sglang={sources.fake_sglang.path}",
                "--build-arg",
                f"DYNAMO_RUNTIME_VERSION={DYNAMO_RUNTIME_VERSION}",
                "--build-arg",
                f"FAKE_SGLANG_COMMIT={sources.fake_sglang.head}",
                "--tag",
                images.serving,
                "--file",
                REPOSITORY_ROOT / "docker" / "serving-prebuilt.Dockerfile",
                REPOSITORY_ROOT,
            ),
            env=docker_env,
        )

    def _build_serving_source(
        self,
        sources: SourcePair,
        images: Images,
        docker_env: dict[str, str],
    ) -> None:
        self.runner.run(
            (
                "docker",
                "buildx",
                "build",
                "--platform",
                "linux/arm64",
                "--load",
                "--target",
                "serving",
                "--build-context",
                f"dynamo={sources.dynamo.path}",
                "--build-context",
                f"fake_sglang={sources.fake_sglang.path}",
                "--build-arg",
                f"DYNAMO_COMMIT={sources.dynamo.head}",
                "--build-arg",
                f"FAKE_SGLANG_COMMIT={sources.fake_sglang.head}",
                "--tag",
                images.serving,
                "--file",
                REPOSITORY_ROOT / "docker" / "serving.Dockerfile",
                REPOSITORY_ROOT,
            ),
            env=docker_env,
        )

    def _build_planner_source(
        self,
        sources: SourcePair,
        images: Images,
        docker_env: dict[str, str],
    ) -> None:
        self.runner.run(
            (
                "docker",
                "buildx",
                "build",
                "--platform",
                "linux/arm64",
                "--load",
                "--target",
                "planner",
                "--build-context",
                f"dynamo={sources.dynamo.path}",
                "--build-context",
                f"fake_sglang={sources.fake_sglang.path}",
                "--build-arg",
                f"DYNAMO_COMMIT={sources.dynamo.head}",
                "--tag",
                images.planner,
                "--file",
                REPOSITORY_ROOT / "docker" / "serving.Dockerfile",
                REPOSITORY_ROOT,
            ),
            env=docker_env,
        )

    def _build_operator_source(
        self,
        sources: SourcePair,
        images: Images,
        docker_env: dict[str, str],
    ) -> None:
        operator_path = Path(sources.dynamo.path) / "deploy" / "operator"
        compliance_path = Path(sources.dynamo.path) / "container" / "compliance"
        if not (operator_path / "Dockerfile").is_file():
            raise RuntimeError(
                "selected DYNAMO_SOURCE cannot build the operator: "
                f"{operator_path / 'Dockerfile'} is missing"
            )
        command: list[str | os.PathLike[str]] = [
            "docker",
            "buildx",
            "build",
            "--platform",
            "linux/arm64",
            "--load",
            "--tag",
            images.operator,
        ]
        if compliance_path.is_dir():
            command.extend(
                ("--build-context", f"compliance={os.fspath(compliance_path)}")
            )
        command.append(operator_path)
        self.runner.run(command, env=docker_env)


def stage_chart(sources: SourcePair) -> Path:
    source_chart = (
        Path(sources.dynamo.path) / "deploy" / "helm" / "charts" / "platform"
    )
    if not (source_chart / "Chart.yaml").is_file():
        raise RuntimeError(
            "selected DYNAMO_SOURCE has no platform Helm chart at "
            f"{source_chart}"
        )
    target = LOCAL_ROOT / "dynamo-platform-chart"
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_chart, target)
    return target
