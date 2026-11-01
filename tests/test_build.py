from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from dynamo_routing_harness.build import ImageBuilder
from dynamo_routing_harness.source import SourceMetadata, SourcePair


class RecordingRunner:
    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []

    def run(self, args, **kwargs):
        command = tuple(os.fspath(arg) for arg in args)
        self.commands.append(command)
        return SimpleNamespace(
            stdout="",
            stderr="",
            returncode=(
                1 if command[:3] == ("docker", "image", "inspect") else 0
            ),
        )


class BuildTests(unittest.TestCase):
    def test_source_build_consumes_both_sources_and_reuses_platform_images(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dynamo = root / "dynamo"
            fake = root / "fake-sglang"
            dynamo.mkdir()
            fake.mkdir()
            sources = SourcePair(
                SourceMetadata(
                    os.fspath(dynamo), "d" * 40, False, (), "1" * 64
                ),
                SourceMetadata(os.fspath(fake), "f" * 40, False, (), "2" * 64),
            )
            runner = RecordingRunner()
            images = ImageBuilder(runner).build(sources)
        serving = next(
            command
            for command in runner.commands
            if command[:3] == ("docker", "buildx", "build")
        )
        self.assertIn(f"dynamo={dynamo}", serving)
        self.assertIn(f"fake_sglang={fake}", serving)
        self.assertIn("DYNAMO_COMMIT=" + "d" * 40, serving)
        self.assertIn("FAKE_SGLANG_COMMIT=" + "f" * 40, serving)
        self.assertEqual(
            images.operator,
            "nvcr.io/nvidia/ai-dynamo/kubernetes-operator:1.6.0",
        )
        self.assertEqual(
            images.planner, "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.6.0"
        )
        build_commands = [
            command
            for command in runner.commands
            if command[:3] == ("docker", "buildx", "build")
        ]
        self.assertEqual(build_commands, [serving])
        loads = [command for command in runner.commands if command[:2] == ("kind", "load")]
        self.assertEqual({command[-1] for command in loads}, {images.serving})

    def test_prebuilt_pulls_ngc_images_and_builds_serving(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "fake-sglang"
            fake.mkdir()
            sources = SourcePair(
                SourceMetadata(
                    "", "", False, (), "1" * 64,
                    requested="ai-dynamo/dynamo@v1.6.0",
                    ref="v1.6.0",
                    remote_url="https://github.com/ai-dynamo/dynamo.git",
                    prebuilt=True,
                ),
                SourceMetadata(os.fspath(fake), "f" * 40, False, (), "2" * 64),
            )
            runner = RecordingRunner()
            images = ImageBuilder(runner).build(sources)
        serving = next(
            command
            for command in runner.commands
            if command[:3] == ("docker", "buildx", "build")
        )
        self.assertIn("serving-prebuilt.Dockerfile", serving[-2])
        self.assertIn(f"fake_sglang={fake}", serving)
        self.assertIn("DYNAMO_RUNTIME_VERSION=1.6.0", serving)
        self.assertNotIn("dynamo=", " ".join(serving))
        self.assertEqual(
            images.operator,
            "nvcr.io/nvidia/ai-dynamo/kubernetes-operator:1.6.0",
        )
        self.assertEqual(
            images.planner, "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.6.0"
        )
        # NGC images are pulled by the nodes, not by the host or kind.
        self.assertFalse(
            [c for c in runner.commands if c[:2] == ("docker", "pull")]
        )
        loads = [c for c in runner.commands if c[:2] == ("kind", "load")]
        self.assertEqual({c[-1] for c in loads}, {images.serving})

    def test_cached_serving_image_skips_rebuild(self) -> None:
        class CachedRunner(RecordingRunner):
            def run(self, args, **kwargs):
                result = super().run(args, **kwargs)
                command = self.commands[-1]
                if command[:3] == ("docker", "image", "inspect"):
                    return SimpleNamespace(stdout="[]", stderr="", returncode=0)
                return result

        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "fake-sglang"
            fake.mkdir()
            sources = SourcePair(
                SourceMetadata(
                    "",
                    "",
                    False,
                    (),
                    "1" * 64,
                    requested="ai-dynamo/dynamo@v1.6.0",
                    ref="v1.6.0",
                    remote_url="https://github.com/ai-dynamo/dynamo.git",
                    prebuilt=True,
                ),
                SourceMetadata(os.fspath(fake), "f" * 40, False, (), "2" * 64),
            )
            runner = CachedRunner()
            images = ImageBuilder(runner).build(sources)

        self.assertFalse(
            [
                command
                for command in runner.commands
                if command[:3] == ("docker", "buildx", "build")
            ]
        )
        loads = [
            command
            for command in runner.commands
            if command[:2] == ("kind", "load")
        ]
        self.assertEqual({command[-1] for command in loads}, {images.serving})


if __name__ == "__main__":
    unittest.main()
