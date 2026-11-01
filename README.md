# Dynamo routing harness

Standalone, Apple Silicon CPU-only lab for comparing Dynamo routing policies
against a **local fake-sglang checkout**. The harness owns orchestration,
validation, metrics, and reporting; fake-sglang remains the serving engine and
Dynamo remains the runtime/operator source.

## Scope

Included:

- Prebuilt vanilla Dynamo by default: the published `ai-dynamo` wheel for the
  serving runtime plus the NGC operator and Planner images. Only the local
  fake-sglang checkout is built from source. Setting `DYNAMO_SOURCE` switches
  the serving runtime to a from-source build (requires a Rust toolchain image);
  the matching release's NGC operator and Planner images remain prebuilt.
- The Helm chart from the pinned vanilla Dynamo release (or the local checkout
  when `DYNAMO_SOURCE` is set).
- A dedicated Colima profile and three-node Kind cluster.
- One frontend, one Planner, four prefill processes with 10 DP ranks each, and
  four decode processes with 10 DP ranks each.
- Mooncake Conversation and ToolAgent workloads under KV-aware, load-aware, and
  PO2C routing (`power-of-two` on the CLI).
- The exact 53-panel `dynamo-sim` Grafana dashboard, including the predicted
  versus observed cache-hit ratio, the complete live metric catalog, raw-series
  exploration, worker/router/Planner views, and infrastructure telemetry.
- Direct Prometheus cache-counter baselines and deltas scoped to the fresh
  prefill pods for each case.
- Strict AIPerf completion, error, output-row, fleet-identity, readiness, and
  restart checks.
- Focused JSON and Markdown reports containing the exact source commits and
  dirty state.

Excluded:

- fake-sglang engine/model implementation.
- Dynamo engine/operator implementation.
- Manual traffic generators, scenario tools, and probe utilities.

The observability path matches the earlier lab: the OpenTelemetry Collector
scrapes Dynamo, fake-sglang, the operator, etcd, NATS, Grafana, Tempo, and Loki,
then exports the combined catalog to Prometheus. Grafana provisions the exact
earlier dashboard plus Prometheus, Tempo, and Loki datasources. Worker and
collector probes, resource bounds, non-root execution, and rollout availability
settings remain enabled. Prometheus recording rules alias semantically equivalent
Dynamo 1.6 router metrics to the Dynamo 1.5 frontend names used by the unchanged
dashboard; panels without a genuine 1.6 equivalent remain empty rather than
displaying fabricated values.

## Source locations

By default Dynamo is the pinned vanilla release (`ai-dynamo` wheel + NGC
operator/planner images); only fake-sglang is a local checkout. Set
`DYNAMO_SOURCE` to build Dynamo from a local Git path or GitHub repository
instead:

```text
workplace/
├── dynamo/        # optional: only when building Dynamo from source
├── fake-sglang/
└── dynamo-routing-harness/
```

```bash
# Dynamo defaults to the prebuilt vanilla release; leave DYNAMO_SOURCE unset.
export FAKE_SGLANG_SOURCE="${FAKE_SGLANG_SOURCE:-../fake-sglang}"
# Opt into a from-source Dynamo build:
# export DYNAMO_SOURCE="../dynamo"
```

Equivalent CLI options are available on every command:

```bash
./bin/lab \
  --dynamo-source ../dynamo \
  --fake-sglang-source ../fake-sglang \
  doctor
```

For remote GitHub sources, use an `owner/repository`, HTTPS, or SSH GitHub
location and optionally select a branch, tag, or commit:

```bash
./bin/lab \
  --dynamo-source ai-dynamo/dynamo \
  --dynamo-ref 701da25292ecca8f64d770cc0b17d389200019a1 \
  --fake-sglang-source karannparikh/fake-sglang \
  --fake-sglang-ref 0ff63336fc0d2061f94b325b6ab5949ecae50c57 \
  up
```

The environment equivalents are `DYNAMO_REF` and `FAKE_SGLANG_REF`. When a
remote ref is omitted, `HEAD` is fetched; use an immutable commit for repeatable
runs. Remote repositories are cloned/fetched with Git into the ignored
`.local/sources/` cache, checked out detached, and rejected if that cache is
dirty or points at a different origin. Existing local checkouts are never
checked out or otherwise mutated.

Local paths must be Git work trees. Every matrix, per-case summary, combined
summary, and Markdown report records the resolved path, full HEAD commit, and
dirty boolean. JSON artifacts also retain status entries and a content
fingerprint, requested location, remote URL, and selected ref.

The two source versions must be mutually compatible. This harness does not
silently replace either checkout with an unrelated `ai-dynamo` package:

- By default `docker/serving-prebuilt.Dockerfile` installs the pinned vanilla
  `ai-dynamo` wheel and layers `FAKE_SGLANG_SOURCE` editable on top; the
  operator and Planner come from the published NGC images at the same version.
- When `DYNAMO_SOURCE` is set, `docker/serving.Dockerfile` instead builds both
  the pure-Python `ai-dynamo` wheel and `ai-dynamo-runtime` native wheel from
  that source. The operator and Planner remain the published NGC 1.6.0 images,
  matching the selected source branch used by this lab.
- Either way the serving image installs `FAKE_SGLANG_SOURCE` editable and
  validates its fake SGLang marker, Dynamo frontend/worker/Planner entry points,
  and effective H200 profile.
- The platform chart is copied from
  `$DYNAMO_SOURCE/deploy/helm/charts/platform` before `helm dependency build`,
  so the selected source checkout is not modified.

Missing build surfaces, native build failures, incompatible Python
dependencies, missing entry points, or contract mismatches fail the image build
with an explicit error.

## Prerequisites

- macOS on Apple Silicon
- Colima, Docker CLI, Kind, kubectl, Helm, Git, and Python 3.11+
- Git access to compatible Dynamo and fake-sglang local checkouts or GitHub
  repositories

The Mooncake Conversation and ToolAgent (agent multi-turn) traces
(`conversation_trace.jsonl` and `toolagent_trace.jsonl`) are the pinned
kvcache-ai/Mooncake FAST'25 traces. `benchmark-setup` downloads and checksum-
verifies them on demand into the git-ignored trace directory when they are
missing, so they are never checked into the repository.

The dedicated Colima profile uses 8 CPUs, 24 GiB RAM, and 100 GiB disk. The Kind
cluster is named `dynamo-routing`; destructive cluster operations verify the
`kind-dynamo-routing` context and loopback API endpoint.

The harness follows the enterprise approved-dependency policy. It installs no
plugins, hooks, MCP integrations, or internet scripts. Container/Python/Helm
dependencies are declared and versioned in repository files.

## Setup and run

```bash
./bin/lab doctor
./bin/lab up
```

`up` builds local-source images for `linux/arm64`, loads them into Kind,
installs the selected local Dynamo chart and operator, deploys the complete
Prometheus/OTel/Grafana/Tempo/Loki/NATS-exporter stack, and validates the graph,
fleet, and effective fake-sglang H200 contract.

The live dashboard is available without a login at:

```text
http://127.0.0.1:3000/d/dynamo-local-sglang
```

Create the pinned AIPerf environment and run the six-case matrix.
`benchmark-setup` downloads the pinned Mooncake traces into the default trace
directory (`.local/mooncake-100rps/`) when they are missing:

```bash
./bin/lab benchmark-setup
./bin/lab benchmark --output reports/mooncake
```

Run a controlled multi-turn KV-return comparison with 1,500 eight-turn
sessions, fixed 256-token user messages, fixed 256-token responses, and 20 RPS:

```bash
./bin/lab benchmark-multiturn \
  --output reports/multiturn-8x256-kv-return \
  --sessions 1500 \
  --turns 8 \
  --input-tokens 256 \
  --output-tokens 256 \
  --request-rate 20 \
  --kv-return both
```

The generated Mooncake-format trace groups turns by `session_id`. AIPerf
captures each live assistant response and includes it in the next request's
conversation history, so the return-on case measures reusable response KV
rather than repeated independent prompts.

Use `--frontend-replicas 4 --no-router-replica-sync` on `deploy`, `benchmark`,
or `benchmark-multiturn` to run four independently tracking Dynamo routers.
The rendered frontend command explicitly disables best-effort router replica
synchronization instead of relying on Dynamo's default.

The default offered load is 20 RPS. Override the rate or point at a different
trace directory without changing the matrix:

```bash
./bin/lab benchmark \
  --request-rate 35 \
  --trace-dir /path/to/mooncake-traces \
  --output reports/mooncake-35rps
```

Other lifecycle commands:

```bash
./bin/lab rebuild
./bin/lab deploy --routing-mode power-of-two
./bin/lab status
./bin/lab report --output reports/mooncake
./bin/lab stop
```

`stop` stops only the dedicated Colima profile.

## Validated contracts

Workers deliberately do **not** pass fake-sglang profile flags. The built image
instantiates fake-sglang's application configuration and requires these
inherited H200 defaults:

| Setting | Required effective value |
|---|---:|
| KV blocks per rank | 15,509 |
| Tokens per page | 256 |
| Token slots per rank | 3,970,304 |
| Context length | 1,048,576 |
| KV cache dtype | FP8 E4M3 |
| Maximum running requests per rank | 32 |

Each case reapplies its routing mode, deletes every graph pod, requires entirely
new UIDs, exact replica/readiness counts, zero initial restarts, correct model
and 10-rank DP arguments, a stable 15-second settle, and unchanged UIDs and
restart counts throughout traffic.

AIPerf must produce the expected number of requests and profiling rows, zero
request errors, an empty error summary, valid request timestamps, and zero
output-length mismatches when that field is present. Cache evidence uses direct
before/after values for both simulated prefix-cache counters across all 40
current prefill ranks; identity changes, counter resets, incomplete labels, or
zero query-token deltas fail the case.

## Output

Each run contains:

```text
matrix.json
summary.json
report.md
<workload>-<routing>/
├── summary.json
├── cache-metrics-before.json
├── cache-metrics-after.json
└── aiperf/
```

Generated cluster state, caches, images metadata, benchmark outputs, and Python
artifacts are ignored by Git.

## Development validation

Unit tests mock Git source checkouts, Kubernetes output, Prometheus responses,
and AIPerf artifacts. They require neither a real Dynamo checkout nor a real
fake-sglang checkout:

```bash
make check
```
