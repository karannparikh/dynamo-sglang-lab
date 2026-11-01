from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from .build import Images, colima_docker_env, stage_chart
from .command import CommandError, Runner
from .constants import (
    CLUSTER_NAME,
    COLIMA_PROFILE,
    DEFAULT_KUBECONFIG,
    DGD_NAME,
    EXPECTED_REPLICAS,
    H200_PROFILE,
    KUBE_CONTEXT,
    LOCAL_ROOT,
    MODEL_NAME,
    MODEL_PATH,
    OBSERVABILITY_NAMESPACE,
    REPOSITORY_ROOT,
    SERVING_NAMESPACE,
    SYSTEM_NAMESPACE,
)
from .manifests import (
    render_grafana_dashboard,
    render_operator_values,
    render_resources,
    write_json_documents,
)
from .source import SourcePair

SELECTOR = f"nvidia.com/dynamo-graph-deployment-name={DGD_NAME}"


class FleetValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class PodState:
    name: str
    uid: str
    role: str
    ready: bool
    phase: str
    restart_count: int
    node: str
    args: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["args"] = list(self.args)
        return value


@dataclass(frozen=True)
class FleetSnapshot:
    pods: tuple[PodState, ...]

    @property
    def uids(self) -> set[str]:
        return {pod.uid for pod in self.pods}

    @property
    def names(self) -> set[str]:
        return {pod.name for pod in self.pods}

    def by_role(self, role: str) -> tuple[PodState, ...]:
        return tuple(pod for pod in self.pods if pod.role == role)

    def as_dict(self) -> dict[str, object]:
        return {
            "pods": [pod.as_dict() for pod in self.pods],
            "roles": {
                role: len(self.by_role(role)) for role in EXPECTED_REPLICAS
            },
        }

    def stable_against(self, baseline: "FleetSnapshot") -> bool:
        current = {
            (pod.name, pod.uid, pod.role, pod.restart_count, pod.ready, pod.phase)
            for pod in self.pods
        }
        expected = {
            (pod.name, pod.uid, pod.role, pod.restart_count, pod.ready, pod.phase)
            for pod in baseline.pods
        }
        return current == expected


def _pod_ready(item: dict[str, object]) -> bool:
    status = item.get("status", {})
    conditions = status.get("conditions", []) if isinstance(status, dict) else []
    return any(
        condition.get("type") == "Ready" and condition.get("status") == "True"
        for condition in conditions
        if isinstance(condition, dict)
    )


def parse_fleet(payload: dict[str, object]) -> FleetSnapshot:
    pods: list[PodState] = []
    for item in payload.get("items", []):
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata", {})
        status = item.get("status", {})
        spec = item.get("spec", {})
        if not all(isinstance(value, dict) for value in (metadata, status, spec)):
            continue
        labels = metadata.get("labels", {})
        role = labels.get("nvidia.com/dynamo-component-type", "")
        containers = spec.get("containers", [])
        statuses = status.get("containerStatuses", [])
        main = next(
            (
                container
                for container in containers
                if isinstance(container, dict) and container.get("name") == "main"
            ),
            containers[0] if containers else {},
        )
        restart_count = sum(
            int(container.get("restartCount", 0))
            for container in statuses
            if isinstance(container, dict)
        )
        pods.append(
            PodState(
                name=str(metadata.get("name", "")),
                uid=str(metadata.get("uid", "")),
                role=str(role).lower(),
                ready=_pod_ready(item),
                phase=str(status.get("phase", "")),
                restart_count=restart_count,
                node=str(spec.get("nodeName", "")),
                args=tuple(str(arg) for arg in main.get("args", [])),
            )
        )
    return FleetSnapshot(tuple(sorted(pods, key=lambda pod: pod.name)))


def validate_fleet(
    snapshot: FleetSnapshot,
    expected_replicas: Mapping[str, int] = EXPECTED_REPLICAS,
) -> None:
    problems: list[str] = []
    for role, expected in expected_replicas.items():
        role_pods = snapshot.by_role(role)
        if len(role_pods) != expected:
            problems.append(f"{role}: expected {expected}, observed {len(role_pods)}")
        for pod in role_pods:
            if pod.phase != "Running" or not pod.ready:
                problems.append(
                    f"{pod.name}: phase={pod.phase!r}, ready={pod.ready!r}"
                )
            if pod.restart_count:
                problems.append(
                    f"{pod.name}: restart_count={pod.restart_count}, expected 0"
                )
    for role in ("prefill", "decode"):
        for pod in snapshot.by_role(role):
            args = list(pod.args)
            if "--dp-size" not in args:
                problems.append(f"{pod.name}: --dp-size is missing")
            else:
                index = args.index("--dp-size")
                value = args[index + 1] if index + 1 < len(args) else ""
                if value != "10":
                    problems.append(f"{pod.name}: --dp-size={value!r}, expected 10")
            if "--enable-dp-attention" not in args:
                problems.append(f"{pod.name}: --enable-dp-attention is missing")
            expected_pairs = {
                "--model-path": MODEL_PATH,
                "--served-model-name": MODEL_NAME,
            }
            for flag, expected_value in expected_pairs.items():
                if flag not in args:
                    problems.append(f"{pod.name}: {flag} is missing")
                    continue
                index = args.index(flag)
                value = args[index + 1] if index + 1 < len(args) else ""
                if value != expected_value:
                    problems.append(
                        f"{pod.name}: {flag}={value!r}, expected {expected_value!r}"
                    )
            forbidden = {
                "--fake-num-kv-blocks",
                "--page-size",
                "--context-length",
                "--kv-cache-dtype",
                "--max-running-requests",
            }
            redundant = sorted(forbidden.intersection(args))
            if redundant:
                problems.append(
                    f"{pod.name}: H200 application defaults were overridden: "
                    + ", ".join(redundant)
                )
    if problems:
        raise FleetValidationError("; ".join(problems))


class Kubernetes:
    def __init__(
        self,
        runner: Runner | None = None,
        kubeconfig: Path = DEFAULT_KUBECONFIG,
    ) -> None:
        self.runner = runner or Runner()
        self.kubeconfig = kubeconfig

    @property
    def env(self) -> dict[str, str]:
        return {"KUBECONFIG": os.fspath(self.kubeconfig)}

    def kubectl(self, *args: str, **kwargs: object):
        env = dict(self.env)
        extra_env = kwargs.pop("env", None)
        if isinstance(extra_env, dict):
            env.update(extra_env)
        return self.runner.run(("kubectl", *args), env=env, **kwargs)

    def require_apple_silicon(self) -> None:
        observed = f"{platform.system()}/{platform.machine()}"
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError(
                "this harness is intentionally Apple Silicon only; "
                f"observed {observed}"
            )

    def require_tools(self) -> None:
        required = ("colima", "docker", "kind", "kubectl", "helm", "git", "python3")
        missing = [tool for tool in required if shutil.which(tool) is None]
        if missing:
            raise RuntimeError("missing required tools: " + ", ".join(missing))

    def _wait_for_nodes(
        self,
        *,
        timeout: float = 120,
        interval: float = 2,
    ) -> dict[str, object]:
        deadline = time.monotonic() + timeout
        last_error = ""
        while time.monotonic() < deadline:
            result = self.kubectl("get", "nodes", "-o", "json", check=False)
            if result.returncode == 0:
                try:
                    payload = json.loads(result.stdout)
                except json.JSONDecodeError as error:
                    last_error = str(error)
                else:
                    if payload.get("items"):
                        return payload
                    last_error = "the API returned no nodes"
            else:
                last_error = result.stderr.strip() or result.stdout.strip()
            time.sleep(interval)
        raise RuntimeError(
            "kind API did not expose its nodes after restart: " + last_error
        )

    def start_cluster(self) -> None:
        self.require_apple_silicon()
        self.require_tools()
        LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
        self.runner.run(
            (
                "colima",
                "start",
                "--profile",
                COLIMA_PROFILE,
                "--arch",
                "aarch64",
                "--cpu",
                "8",
                "--memory",
                "24",
                "--disk",
                "100",
                "--vm-type",
                "vz",
                "--mount-type",
                "virtiofs",
            )
        )
        clusters = self.runner.run(
            ("kind", "get", "clusters"), env=colima_docker_env()
        ).stdout.splitlines()
        if CLUSTER_NAME not in clusters:
            self.runner.run(
                (
                    "kind",
                    "create",
                    "cluster",
                    "--name",
                    CLUSTER_NAME,
                    "--config",
                    REPOSITORY_ROOT / "config" / "kind.yaml",
                    "--kubeconfig",
                    self.kubeconfig,
                    "--image",
                    "kindest/node:v1.35.0",
                ),
                env=colima_docker_env(),
            )
        else:
            kubeconfig = self.runner.run(
                ("kind", "get", "kubeconfig", "--name", CLUSTER_NAME),
                env=colima_docker_env(),
            ).stdout
            self.kubeconfig.parent.mkdir(parents=True, exist_ok=True)
            self.kubeconfig.write_text(kubeconfig)
        context = self.kubectl("config", "current-context").stdout.strip()
        endpoint = self.kubectl(
            "config", "view", "--minify", "-o", "jsonpath={.clusters[0].cluster.server}"
        ).stdout.strip()
        if context != KUBE_CONTEXT or not endpoint.startswith("https://127.0.0.1:"):
            raise RuntimeError(
                "refusing to manage an unexpected cluster: "
                f"context={context!r}, endpoint={endpoint!r}"
            )
        nodes = self._wait_for_nodes()
        workers = [
            item["metadata"]["name"]
            for item in nodes["items"]
            if "node-role.kubernetes.io/control-plane"
            not in item["metadata"].get("labels", {})
        ]
        if len(workers) != 2:
            raise RuntimeError(f"expected 2 kind workers, observed {len(workers)}")
        for node in workers:
            self.kubectl(
                "patch",
                "node",
                node,
                "--subresource=status",
                "--type=merge",
                "-p",
                json.dumps(
                    {
                        "status": {
                            "capacity": {"nvidia.com/gpu": "10"},
                            "allocatable": {"nvidia.com/gpu": "10"},
                        }
                    }
                ),
            )
        observed_nodes = json.loads(
            self.kubectl("get", "nodes", "-o", "json").stdout
        )
        for item in observed_nodes["items"]:
            labels = item["metadata"].get("labels", {})
            if labels.get("dynamo-sim.local/gpu-capacity") != "simulated":
                continue
            allocatable = item["status"].get("allocatable", {})
            if allocatable.get("nvidia.com/gpu") != "10":
                raise RuntimeError(
                    f"{item['metadata']['name']} does not advertise 10 simulated GPUs"
                )

    def install_platform(
        self,
        sources: SourcePair,
        images: Images,
    ) -> None:
        chart = stage_chart(sources)
        values = LOCAL_ROOT / "operator-values.yaml"
        render_operator_values(
            images.operator_repository,
            images.operator_tag,
            values,
        )
        self.runner.run(("helm", "dependency", "build", chart), env=self.env)
        self.runner.run(
            (
                "helm",
                "upgrade",
                "--install",
                "dynamo-platform",
                chart,
                "--namespace",
                SYSTEM_NAMESPACE,
                "--create-namespace",
                "--values",
                values,
                "--wait",
                "--timeout",
                "10m",
            ),
            env=self.env,
        )

    def install_observability(self) -> None:
        self.kubectl(
            "apply",
            "-f",
            "-",
            input_text=json.dumps(
                {
                    "apiVersion": "v1",
                    "kind": "Namespace",
                    "metadata": {"name": OBSERVABILITY_NAMESPACE},
                }
            ),
        )
        self.kubectl(
            "apply",
            "-f",
            os.fspath(REPOSITORY_ROOT / "config" / "nats-exporter.yaml"),
        )
        for name in (
            "tempo.yaml",
            "loki.yaml",
            "otel-collector.yaml",
            "prometheus.yaml",
        ):
            self.kubectl(
                "apply",
                "-f",
                os.fspath(REPOSITORY_ROOT / "config" / name),
            )
        dashboard_manifest = LOCAL_ROOT / "grafana-dashboard-configmap.json"
        write_json_documents(
            [render_grafana_dashboard()],
            dashboard_manifest,
        )
        self.kubectl(
            "apply",
            "--server-side",
            "--force-conflicts",
            "-f",
            os.fspath(dashboard_manifest),
        )
        self.kubectl(
            "apply",
            "-f",
            os.fspath(REPOSITORY_ROOT / "config" / "grafana.yaml"),
        )
        self.kubectl(
            "-n",
            SYSTEM_NAMESPACE,
            "rollout",
            "restart",
            "deployment/nats-exporter",
        )
        self.kubectl(
            "-n",
            SYSTEM_NAMESPACE,
            "rollout",
            "status",
            "deployment/nats-exporter",
            "--timeout=5m",
        )
        deployments = ("tempo", "loki", "otel-collector", "prometheus", "grafana")
        for deployment in deployments:
            self.kubectl(
                "-n",
                "observability",
                "rollout",
                "restart",
                f"deployment/{deployment}",
            )
            self.kubectl(
                "-n",
                "observability",
                "rollout",
                "status",
                f"deployment/{deployment}",
                "--timeout=5m",
            )

    def deploy_graph(
        self,
        image: str,
        planner_image: str,
        sources: SourcePair,
        routing_mode: str = "kv",
        enable_kv_return: bool = False,
        fake_output_token_id: int | None = None,
        frontend_replicas: int = 1,
        router_replica_sync: bool = False,
    ) -> Path:
        return_suffix = "-kv-return" if enable_kv_return else ""
        token_suffix = (
            f"-token-{fake_output_token_id}"
            if fake_output_token_id is not None
            else ""
        )
        frontend_suffix = (
            f"-frontends-{frontend_replicas}"
            if frontend_replicas != 1
            else ""
        )
        sync_suffix = "-router-sync" if router_replica_sync else "-no-router-sync"
        manifest = (
            LOCAL_ROOT
            / (
                f"dgd-{routing_mode}{return_suffix}{token_suffix}"
                f"{frontend_suffix}{sync_suffix}.json"
            )
        )
        write_json_documents(
            render_resources(
                image,
                planner_image,
                sources,
                routing_mode,
                enable_kv_return,
                fake_output_token_id,
                frontend_replicas,
                router_replica_sync,
            ),
            manifest,
        )
        self.kubectl("apply", "-f", os.fspath(manifest))
        return manifest

    def wait_graph(self, timeout: float = 600) -> None:
        deadline = time.monotonic() + timeout
        last = ""
        while time.monotonic() < deadline:
            result = self.kubectl(
                "-n",
                SERVING_NAMESPACE,
                "get",
                "dgd",
                DGD_NAME,
                "-o",
                "json",
                check=False,
            )
            if result.returncode == 0:
                payload = json.loads(result.stdout)
                generation = payload.get("metadata", {}).get("generation")
                status = payload.get("status", {})
                observed = status.get("observedGeneration")
                conditions = status.get("conditions", [])
                ready = any(
                    condition.get("type") == "Ready"
                    and condition.get("status") == "True"
                    for condition in conditions
                )
                last = json.dumps(
                    {
                        "generation": generation,
                        "observedGeneration": observed,
                        "ready": ready,
                        "state": status.get("state"),
                    }
                )
                if ready and observed == generation:
                    return
            time.sleep(5)
        raise TimeoutError(f"DynamoGraphDeployment did not become ready: {last}")

    def fleet(self) -> FleetSnapshot:
        payload = json.loads(
            self.kubectl(
                "-n",
                SERVING_NAMESPACE,
                "get",
                "pods",
                "-l",
                SELECTOR,
                "-o",
                "json",
            ).stdout
        )
        return parse_fleet(payload)

    def wait_fleet(
        self,
        *,
        old_uids: set[str] | None = None,
        timeout: float = 600,
        expected_replicas: Mapping[str, int] = EXPECTED_REPLICAS,
    ) -> FleetSnapshot:
        deadline = time.monotonic() + timeout
        last_error = "fleet not observed"
        while time.monotonic() < deadline:
            try:
                snapshot = self.fleet()
                validate_fleet(snapshot, expected_replicas)
                if old_uids and snapshot.uids.intersection(old_uids):
                    raise FleetValidationError(
                        "new fleet reused pod UIDs from the previous case"
                    )
                return snapshot
            except (CommandError, FleetValidationError, json.JSONDecodeError) as error:
                last_error = str(error)
                time.sleep(5)
        raise TimeoutError(f"fleet did not stabilize: {last_error}")

    def recreate_fleet(
        self,
        previous_uids: set[str],
        expected_replicas: Mapping[str, int] = EXPECTED_REPLICAS,
    ) -> FleetSnapshot:
        self.kubectl(
            "-n",
            SERVING_NAMESPACE,
            "delete",
            "pods",
            "-l",
            SELECTOR,
            "--wait=true",
            "--timeout=3m",
        )
        return self.wait_fleet(
            old_uids=previous_uids,
            expected_replicas=expected_replicas,
        )

    def validate_profile_contract(self, snapshot: FleetSnapshot) -> dict[str, object]:
        prefill = snapshot.by_role("prefill")
        if not prefill:
            raise FleetValidationError("no prefill pod available for profile validation")
        payload = self.kubectl(
            "-n",
            SERVING_NAMESPACE,
            "exec",
            prefill[0].name,
            "-c",
            "main",
            "--",
            "cat",
            "/opt/harness/effective-profile.json",
        ).stdout
        contract = json.loads(payload)
        if contract.get("profile") != H200_PROFILE:
            raise FleetValidationError(
                "runtime H200 profile contract does not match the harness contract"
            )
        if contract.get("profile_inherited") is not True:
            raise FleetValidationError("runtime profile is not inherited from fake-sglang")
        return contract

    def status(self) -> str:
        parts = [
            self.kubectl(
                "-n", SERVING_NAMESPACE, "get", "dgd,pods", "-o", "wide", check=False
            ).stdout,
            self.kubectl(
                "-n", "observability", "get", "pods", "-o", "wide", check=False
            ).stdout,
            self.kubectl(
                "-n",
                SYSTEM_NAMESPACE,
                "get",
                "pods",
                "-l",
                "app=nats-exporter",
                "-o",
                "wide",
                check=False,
            ).stdout,
        ]
        return "\n".join(part.rstrip() for part in parts if part.strip()) + "\n"

    def stop(self) -> None:
        self.runner.run(
            ("colima", "stop", "--profile", COLIMA_PROFILE),
            check=False,
        )


class PortForward:
    def __init__(
        self,
        kubeconfig: Path,
        namespace: str,
        resource: str,
        ports: str,
    ) -> None:
        self.kubeconfig = kubeconfig
        self.namespace = namespace
        self.resource = resource
        self.ports = ports
        self.process: subprocess.Popen[str] | None = None

    def __enter__(self) -> "PortForward":
        env = os.environ.copy()
        env["KUBECONFIG"] = os.fspath(self.kubeconfig)
        self.process = subprocess.Popen(
            (
                "kubectl",
                "-n",
                self.namespace,
                "port-forward",
                self.resource,
                self.ports,
            ),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(2)
        if self.process.poll() is not None:
            _, stderr = self.process.communicate()
            raise RuntimeError(
                f"port-forward {self.resource} exited early: {stderr.strip()}"
            )
        return self

    def __exit__(self, *args: object) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
