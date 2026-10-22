# Dynamo SGLang Lab

A reproducible Apple Silicon lab for distributed inference experiments: Colima,
a three-node kind Kubernetes cluster, NVIDIA Dynamo and its operator/Planner,
a CPU-only SGLang simulator, observability, dashboards, and traffic tooling.
The complete local infrastructure setup is in `deploy/local/`; this repository
is not just the simulator.

The included CPU-only **fake SGLang engine** simulates prefill, decode, and KV-cache
transfer, so you can develop and test a distributed inference framework on
[NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) + SGLang without GPUs.

This lab builds on the Apache-2.0
[fake-sglang project](https://github.com/sansri61/fake-sglang), with local
infrastructure, simulator extensions, and benchmark tooling. Upstream licenses
and the bundled tokenizer's provenance are retained. Generated `.local/` state,
kubeconfigs, credentials, VM disks, and benchmark outputs are not distributed.

It works by shipping an importable `sglang` package that implements the subset
of SGLang's Python API that Dynamo's SGLang backend consumes. **Dynamo runs
completely unmodified** — `python -m dynamo.sglang` loads this instead of real
SGLang and never knows the difference.

> This is **not** SGLang and does no inference. Output tokens are deterministic
> noise; latencies come from a model, not a forward pass. Every worker prints a
> `SIMULATED engine` banner at startup for exactly this reason.

## Why not something that already exists

| Existing | What it gives | Why it wasn't enough |
|---|---|---|
| `python -m dynamo.mocker` | Dynamo's Rust scheduler simulator: KV blocks, chunked prefill, preemption, bandwidth-modelled transfer | Plugs in as `EngineType.Mocker`. The `dynamo.sglang` component never loads, so none of the SGLang integration is exercised |
| `dynamo.common.backend.sample_engine` | A CPU fake engine on Dynamo's Unified Backend SDK | Same: it bypasses `dynamo.sglang` entirely |
| `sglang/tools/sglang-simulator` | Real SGLang scheduler with a mocked `ModelRunner.forward`, GPU-free in CI | **No PD-disaggregation support at all**, and it still needs the full CUDA-only `sglang` + `torch` install |

## What it simulates

- **Prefill** — chunked against a token budget, cost proportional to tokens
  actually recomputed (prefix-cache hits and transferred KV are free).
- **Decode** — one shared step per batch whose cost grows with batch size, so
  queueing shows up as ITL growth the way it does on a GPU.
- **KV cache** — a finite pool of fixed-size blocks, cumulative-hash prefix
  reuse, and LIFO preemption under pressure.
- **KV transfer** — a **real** cross-process HTTP bootstrap handshake between
  the prefill and decode workers, with the payload cost simulated as
  `tokens x kv_bytes_per_token / bandwidth`. `kv_bytes_per_token` is derived
  from the model's real `config.json`.
- **KV events** — block stores and evictions published over ZMQ in SGLang's
  exact wire format, so Dynamo's KV-aware router builds a real prefix tree and
  routes on cache hits.
- **Scheduler metrics** — pushed over ZMQ to the socket Dynamo binds, so
  load-based routing sees real occupancy.
- **Forward-pass metrics (FPM)** — simulated scheduled/queued token counts,
  batch sizes, variances and pass durations, relayed through real Dynamo.
  Idle heartbeats clear stale load so the Planner can scale down.

Aggregated and disaggregated modes produce **byte-identical output** for the
same prompt, and both are reproducible run to run.

## Local Kubernetes lab (Apple Silicon)

This is the recommended DC-style setup on a Mac. It runs the released
**Dynamo 1.5.0 frontend, SGLang adapter, operator and Planner unchanged**, with
only SGLang's engine replaced by this CPU simulator.

```text
OpenAI streaming traffic
  -> Dynamo frontend / KV-aware router
  -> dynamo.sglang prefill -> simulated KV handoff -> dynamo.sglang decode

Kubernetes discovery + worker metadata <-> Dynamo operator / Planner
NATS JetStream <-> Dynamo events, KV events and forward-pass telemetry

Service /metrics -> OpenTelemetry Collector -> Prometheus -> Grafana
Dynamo OTLP traces/logs -> Collector -> Tempo / Loki -> Grafana
```

The dedicated Colima profile and three-node kind cluster are both named
`dynamo-sim`. They use 8 virtual CPUs, 24 GiB RAM and a 100 GiB virtual disk.
Existing Docker/Kubernetes contexts are not activated or overwritten.
Standalone etcd is also deployed and monitored, but active Dynamo discovery
uses Kubernetes: its worker metadata supplies the Planner's KV capacity.

### Start or resume

Install prerequisites through Homebrew if needed:

```bash
brew install colima docker docker-buildx kind helm kubernetes-cli
```

Clone the lab and NVIDIA Dynamo into sibling directories:

```bash
git clone https://github.com/karannparikh/dynamo-sglang-lab.git
git clone --branch v1.5.0 --depth 1 https://github.com/ai-dynamo/dynamo.git
cd dynamo-sglang-lab
```

The sibling `../dynamo` checkout must be NVIDIA's `v1.5.0` release.
Alternatively, set `DYNAMO_SOURCE`
to an existing checkout at that release; the lab never changes its source.

From this repository:

```bash
bash deploy/local/lab.sh up
bash deploy/local/lab.sh status
```

The first run builds a native ARM64 Linux image containing genuine Dynamo
wheels and explicitly CPU-only PyTorch. Subsequent runs reuse it. No weights
or remote model code are downloaded.

| Endpoint | URL |
|---|---|
| Inference API | http://127.0.0.1:8000 |
| Full metrics dashboard | http://127.0.0.1:3000/d/dynamo-local-sglang |
| Prometheus | http://127.0.0.1:9090 |
| OTLP gRPC | `127.0.0.1:4317` |
| OTLP HTTP | http://127.0.0.1:4318 |

Grafana's username is `dynamo`. Its generated password is in
`.local/grafana-password` (mode 0600). `.local/` is ignored by Git.
Host ports bind only to localhost. This is a development lab, not a
production-hardened deployment; the operator metrics scrape uses bearer
authentication but skips verification of its locally generated certificate.

### Generate traffic

```bash
bash deploy/local/lab.sh traffic --requests 32 --concurrency 8 \
  --prompt-words 512 --max-tokens 128

# Sustained pressure at the fixed worker count:
bash deploy/local/lab.sh traffic --requests 180 --concurrency 32 \
  --prompt-words 2048 --max-tokens 512 --rate 4
```

The client prints JSON lines with per-request latency, client-observed TTFT,
actual token usage and a final success/error summary. It exits nonzero on
HTTP, stream or missing-usage failures. `--prompt-words` controls text length,
not exact token count; actual input tokens are reported. `--prefix-groups`
controls shared-prefix groups. `--rate` is the target offered rate, subject
to the concurrency limit. A stream chunk is **not necessarily one token**;
use the Dynamo histogram for inter-token latency.

The lab exposes **40 prefill + 40 decode logical workers**, packed into four
Python processes per role with ten data-parallel (DP) ranks per process.
Kubernetes discovery registers the process endpoints and their routable rank
ranges. Dynamo routes to `(worker_id, dp_rank)`; each rank has its own scheduler,
KV pool, cache events, scheduler load, and forward-pass metrics. Runtime imports
and model metadata are shared within the process. The real Planner still receives
FPM and evaluates load, but its four-endpoint minimum per role and eight-slot
GPU budget pin the fleet at eight processes. The budget counts the single
simulated GPU scheduling resource per process, not its ten logical ranks.
`optimization_target=load` always enables load scaling in Dynamo, so disabling
its boolean flag alone would not freeze the fleet.

Each worker node advertises ten **simulated Kubernetes GPU resource slots**;
each worker process requests one. These are opaque scheduling resources, not
physical CUDA devices. `up` re-advertises them after node/VM restarts.
This lets the real operator and Planner consume positive resource shapes
without modifying Dynamo's GPU validation.

The dedicated Colima VM uses the supported gRPC port forwarder. This avoids
sharing high-concurrency inference connections with the SSH forwarding
connection used by older lab setups. The initial SSH-based benchmark lost all
host-side forwarded services near 1,006 client requests in flight while the
guest services remained running; the exact SSH failure mechanism is unconfirmed.

### Configure cache capacity and K/V-generation time

Edit `deploy/local/lab-config.json`, then run
`bash deploy/local/lab.sh deploy`. These settings apply to each fake prefill
and decode worker; redeployment restarts affected workers and clears their
caches.

| Setting | Local default | Meaning |
|---|---|---|
| `worker_processes_per_role` | 4 | Kubernetes worker pods per role |
| `ranks_per_process` | 10 | Independent simulated workers per pod |
| `context_length` | 131072 | Simulated maximum sequence length; not a real-model capability claim |
| `kv_blocks_per_worker` | 8192 | Finite KV pool per worker |
| `kv_block_size` | 16 | Token slots per block; default capacity is 131,072 tokens |
| `publish_kv_events` | `true` | Publish retained prefixes and evictions to real Dynamo |
| `kv_generation_ms_per_1k_tokens` | 50.0 | Additional modeled K/V-generation cost per 1,000 uncached tokens |

The context and pool accommodate the full Mooncake FAST'25 conversation and
tool/agent traces (maximum input plus output: 126,527 tokens). The pool stores
block metadata, not GPU KV tensors. Longer requests and more cached blocks
still increase host-memory use; 131,072 slots are not a guarantee that every
concurrent request can fit. The tokenizer/model configuration remains Qwen3-0.6B;
the larger context is an explicit simulation-only override.

The delay is **additive** to the existing prefill cost and overhead, divided
by `fake_speedup_ratio`. It models time, not actual key/value arrays or GPU
computation. Reused prefix tokens and transferred decode KV incur no
K/V-generation cost. Recomputed tokens after preemption are charged again.
The engine's standalone default is zero; the lab deliberately sets 50 ms.
`sglang:simulated_kv_generation_seconds_total` accumulates this modeled time,
excluding health checks/warmups, and has its own Grafana panel.

Capacity, occupancy and prefix changes travel through the existing genuine
Dynamo/SGLang metadata, scheduler metrics and KV-event integration. Turning
publication off does not disable the engine's cache or its observed-hit
metrics. Both settings keep event-driven routing enabled; they do not turn
on Dynamo's approximate prediction of cache state from previous routes.

### AIPerf Mooncake replay against the deployed lab

`scripts/mooncake_lab.py` drives the actual local inference endpoint, not an
isolated replacement engine. It requires the approved AIPerf environment at
`.local/aiperf-venv` (versions in `deploy/local/aiperf-requirements.txt`), the
Qwen3-0.6B tokenizer revision `c1899de289a04d12100db370d81485cdf75e47ca`
cached under `.local/aiperf-hf`, and the complete Mooncake FAST'25 files
`conversation_trace.jsonl` and `toolagent_trace.jsonl` under
`.local/mooncake-100rps`.

```bash
python3 scripts/mooncake_lab.py \
  --request-rate 50 \
  --output ".local/mooncake-100rps/comparison-$(date +%Y%m%d-%H%M%S)"

python3 scripts/mooncake_report.py .local/mooncake-100rps/<comparison-directory>
```

The runner replays both traces separately at a constant offered 50 requests/s
(configurable with `--request-rate`),
once with KV-aware routing and once with Dynamo's `--load-aware` mode.
Use `--modes kv load-aware power-of-two` to also replay both traces with
power-of-two-choices (Po2C), or `--modes power-of-two` for that comparison alone.
Po2C selects between two worker processes using in-flight request counts; it
does not score individual DP ranks or cache overlap. Unpinned requests are
distributed across the fake engine's ranks locally. Dynamo uses its synchronous
prefill fallback in this mode, so this compares complete routing modes, not only
their worker-selection formulas. KV-specific routing controls do not apply.
Original trace timestamps are replaced; request order, lengths, and shared
prefix hashes come from the full trace. Mooncake's prefix block size is
512 tokens, distinct from the fake engine's 16-token cache pages. Chat-template
wrapping adds tokens (eight in the three-request preflight); server-reported
token counts are retained. Tool/agent traces represent inference traffic,
not real tool execution or a quality evaluation.

Each case restarts the frontend and all configured workers for cold caches.
Worker caching remains enabled in every mode.
Each benchmark request sets `ignore_eos: true` so synthetic EOS token IDs do not
end generation before the trace's requested output length. AIPerf still records
actual token counts and output-length mismatches.
The runner temporarily changes
the live DGD routing arguments and restores the original arguments on exit;
do not run competing lab deployments during a comparison. The result directory
records the effective deployment, trace checksum, exact AIPerf command,
per-request results, and five-second kubelet CPU/memory snapshots. CPU samples
use cumulative **container** counters; all three kind nodes share the same
8-vCPU VM and their node-level CPU values must not be added together.
If a container's cumulative CPU counter decreases within a report window,
the affected role's CPU aggregate is marked unavailable with an explicit
warning and container identity; request results and unaffected roles remain
reportable.
The collector scrapes Dynamo pods only after Kubernetes reports them Ready.
This avoids a reproduced startup interaction between synchronous FPM relay
construction (holding Python's GIL) and concurrent Python metrics callbacks.
It is a lab-side mitigation, not a fix to Dynamo's Rust bindings; unready pods
do not export scraped metrics through this collector.
Host AIPerf process-tree resource usage is recorded by `/usr/bin/time -l`.
Dataset preparation has a 900-second timeout (960-second service envelope)
because synthesizing the complete agent trace exceeded AIPerf's default
300-second preparation timeout on the local host. Request timing still starts
only after preparation; the offered rate and per-request timeout are unchanged.
The report separates CPU during request dispatch from CPU including draining,
uses kubelet sample timestamps rather than polling intervals, and records
coverage fractions. A fraction below one indicates incomplete CPU coverage,
not idle CPU. One core corresponds to 100% CPU; worker-role totals aggregate
all workers in that role, using replica counts saved with each case.

New runs also record per-role peak container working-set memory and prefill
cache-hit token fractions, excluding retired pods and decode-side transferred
KV. Fleet identity/restart checks flag invalid cold-counter comparisons. Memory
totals use contemporaneous samples from all processes, not a sum of each
process's independent peak. These CPU timings and memory costs describe the
packed CPU simulator, not 80 operating-system processes or real GPU capacity.

The initial admission settings in `deploy/local/router-policy.yaml` use FCFS,
a prefill busy threshold of 8,192 active tokens, and these queue budgets:

| Limit | Per process (ten ranks) | Four processes per role |
|---|---:|---:|
| Queued requests | 100 | 400 |
| Queued raw input tokens | 655,360 | 2,621,440 |

These budgets apply **per routing queue/policy class**, not globally across
frontend and decode-side router instances. Dynamo checks existing usage before
enqueueing, so the token budget can be exceeded by one accepted request.
Requests already executing are excluded. The policy file is a renderer template:
Dynamo scales these budgets by discovered **process endpoints**, so the renderer
multiplies the per-logical-worker budgets by `ranks_per_process`.
Each rank has eight running scheduler slots. Dynamo's process-wide admission
budget is 80 engine requests plus 20 overflow requests for ten packed ranks;
overflow is shared, not reserved as two slots for each rank. The frontend also uses an
85% active-decode-block threshold with output-block tracking.
These are initial bounded lab settings, not production-calibrated limits.
Full offered request counts include overload errors and timeouts; successful
throughput must not be mistaken for the configured offered rate.

`scripts/packed_probe.py --targets-per-role 40 --output .local/packed-probe.json`
checks all 80 targets through the live frontend, verifies rank-local cache
isolation and token counts, and confirms that unpinned KV-aware routing chooses
a rank holding an exclusive cached prefix. Its explicit routing hints are only
for validation; the Mooncake comparison does not pin requests or sessions.

### Deterministic cache-routing comparison

```bash
bash deploy/local/lab.sh routing-test

# Shorter run or a different simulated K/V-generation cost:
bash deploy/local/lab.sh routing-test --pairs 1 --requests 12 \
  --kv-generation-ms 500
```

The command creates a bounded, isolated test pod in the dedicated lab and
removes it when finished. Logs and the detailed JSON report remain in
`.local/routing-<timestamp>-<suffix>/`. The experiment uses the built lab
image; run `bash deploy/local/lab.sh rebuild` after source changes.

Each successful run also publishes a **separate recorded A/B dashboard** and
prints a link with fixed absolute `from`/`to` times. Sky-blue series are metadata
ON and orange series are OFF (color-blind-friendly Okabe-Ito colors).
Condition labels and line styles provide non-color cues. Every replay point uses its actual request
completion timestamp; phase regions and eviction markers identify the test
windows. The time axis follows the browser timezone. Setup/teardown gaps are
left empty rather than filled with zeroes.

Both dashboard generators always include a top row with **overall QPS**,
**prefill QPS by worker**, and **decode QPS by worker** in three separate
panels. The recorded dashboard counts replay completions in non-overlapping
one-second buckets anchored at the first replay request's start in each case.
The final partial bucket is normalized by its actual duration; an empty
bucket within replay is zero. Setup/teardown windows are not filled in.
Worker counts are attributed using the selected destinations at the request's
E2E completion timestamp, not the unrecorded engine-phase timestamps.
Each role's workers sum to overall QPS; adding both roles would double-count
queries. Seeds and eviction checks are excluded.

The live dashboard uses Prometheus rates: overall QPS counts completed
frontend requests (including error outcomes); the worker panels count
arrivals at the `generate` endpoint, filtered by prefill/decode role.
Management and cache-query endpoints are excluded. Retries/canaries using
`generate` remain included. These arrival and completion rates need not
match instantaneously, and the scrape-based averaging window differs from
the recorded dashboard's one-second buckets.

For a 10-minute replay in **each** of the four ON/OFF/OFF/ON cases:

```bash
bash deploy/local/lab.sh routing-test --duration-seconds 600
```

Duration mode replays the same 600 deterministic requests at one request per
second in every case, serially, then runs the explicit pressure/relocation
check. This is 40 minutes of replay plus setup/teardown, not a saturation
benchmark. The pod deadline expands to cover the run. The final recorded
dashboard is published when all cases finish. Default short runs still use
24 requests per case. Duration mode's QPS includes idle time through the
end of the replay window; slower-than-offered service can extend the window.

For the concurrent, changing-locality workload at **100 offered requests/s**:

```bash
bash deploy/local/lab.sh routing-test --duration-seconds 600 --rate 100 --concurrency 256
```

Each case schedules 60,000 arrivals. A seeded 80/20 mix selects four hot prefix
families or the remaining 28 families; the hot set rotates every two minutes.
Prefixes vary between 256 and 1,024 tokens, plus a 16-token suffix, with eight
output tokens. These are synthetic locality assumptions, not a production
trace. Cache capacity and simulated compute settings remain unchanged.

The generator is open-loop with a bounded in-flight limit: arrivals exceeding
that limit are counted as **client rejections**, never silently slowed down
or mislabeled as server failures. Charts distinguish offered/sent/successful/
failed QPS, worker generate-endpoint arrival rates, client in-flight backlog,
request-start lag, and successful-response latency percentiles. Cases may
include drain time after the ten-minute arrival window.
When overloaded, cases can admit different subsets of the same offered
sequence. This is a saturation comparison, not a matched set of completed
requests; interpret cache ratios alongside admission and completion counts.

Known-owner checks before load and pressure/relocation checks after it remain
serial and are excluded from the concurrent load metrics. Engine counters are
sampled about once a second; concurrent requests cannot be validated using
per-request differences of shared counters. The schema-v3 report stores
one-second client aggregates and worker snapshots, rather than hundreds of
thousands of individual curves. Identical successful query outputs are checked
within/across cases; errors and rejected offered requests are reported separately.

The concurrent recorded dashboard distinguishes **whole-case latency** from
**one-second latency percentiles**. Below the QPS panels, a shared comparison
table shows each case's successful-request count, TTFT p95, E2E p95 and their
difference, plus request-level pooled ON/OFF summaries. Beside it, solid TTFT
and dashed E2E lines show the saved whole-case p95 as fixed references across
each load window. These summaries include drain, exclude setup/eviction checks,
and do not change when zooming. The difference is **E2E p95 - TTFT p95**, not
the p95 of per-request decoding duration.

The separate one-second p50/p95/p99 charts remain available for short-window
variation. With few completions per second, p95 and p99 often equal that
second's maximum. Averaging those bucket percentiles does not recover the
whole-case percentile. Cases with no successful responses show no latency,
not zero. Existing schema-v3 reports already contain the whole-case and pooled
statistics, so these panels can be published without rerunning inference.

### Cache size and eviction graphs

While an experiment is running, open
`http://127.0.0.1:3000/d/dynamo-routing-live` for live QPS, latency, prefix reuse,
cache size and eviction graphs. Select the experiment pod in the run selector.
The collector discovers all four internal metrics endpoints without changing
the experimental worker processes. Live data starts when scraping is enabled;
it cannot backfill earlier observations. Worker slots keep the two prefill
processes distinct even though they share a pod and DP rank.

The live view refreshes every five seconds and includes setup/validation
requests. It does not expose client offered/rejected counts; use the final
recorded dashboard for those and for precise case annotations. Live colors
identify worker slots/statistics, not metadata ON/OFF. Recorded dashboards
remain immutable measurements, not live subscriptions; choosing a recent
time range that excludes their saved run will show no points.

New recordings include shared graphs for resident/allocated/capacity tokens
across all workers, per-worker resident tokens and capacity, and cumulative
pressure-evicted tokens. Samples include seed, replay, pressure and relocation
requests at the actual metric-observation timestamps. Pressure regions are
marked separately. Old reports explicitly show that occupancy history was not
recorded; it cannot be recovered from hit ratios.

The A/B cache has **96 blocks x 16 = 1,536 token slots per worker**:
3,072 across two prefill workers and 1,536 on decode. The main lab config
uses 8,192 blocks x 16 = 131,072 per worker; fleet capacity varies with replicas.
These are simulated token slots, not bytes or unique tokens across replicas.

Engine metrics (all carry `dp_rank` and Dynamo worker labels):

| Metric suffix under `sglang:simulated_prefix_cache_` | Meaning |
| --- | --- |
| `capacity_tokens` | Configured KV token slots |
| `resident_tokens` | Reusable full-block prefix tokens, referenced or unreferenced |
| `allocated_tokens` | Occupied slots, including active/partial blocks |
| `evictable_tokens` | Unreferenced prefix tokens; parent dependencies still apply |
| `evicted_tokens_total` | Tokens evicted by allocation pressure |
| `cleared_tokens_total` | Tokens removed by explicit cache flushes |

Eviction is **space-based LRU, with no TTL**. Allocation consumes clean blocks
first, then evicts unreferenced LRU leaves; active references protect blocks.
Metrics update on pool changes even with KV publication disabled, making
ON/OFF eviction comparisons valid. Eviction counts are token turnover and can
exceed capacity. More eviction is not inherently better: compare pressure
with cache reuse and latency. Post-request samples omit transient peaks;
decode's transferred KV can already be freed at observation time.

One shared panel shows all prefill workers' running token-weighted hit
ratios: reused tokens divided by queried tokens for each worker's replay
requests, reset for each case. Each series labels its worker and case;
blue/orange distinguishes metadata ON/OFF and solid/dashed lines distinguish
workers. Worker numbers are logical aliases for the
fresh processes in each case. Seeds, eviction checks and transferred decode
KV are excluded; an unselected worker has no series rather than a false zero.

For serial (schema-v2) reports, E2E latency is shown both per request and as running **p50/p95/p99** across
workers within each case, reset at the next case. The overview also shows
percentiles pooled across all replay requests for each metadata condition;
it does not average per-case percentiles. p50 uses the median; p95 and p99
use nearest rank. With the default 24 requests per case (48 per condition),
p99 is simply the observed maximum, not a statistically stable tail estimate.
These aggregates are calculated from saved `latency_ms` observations, so
existing timestamped reports can be republished without rerunning traffic.

The dashboard embeds the measured report values as CSV using Grafana's
built-in TestData datasource **in CSV-content mode only**; it generates no
synthetic samples. This avoids losing short experiments between scrapes and
keeps test data separate from the live serving metrics. Dashboards and
annotations persist in Grafana's existing volume. A run's `dashboard.json`
and `dashboard-link.json` are saved next to its report; the live lab's
**Recorded A/B experiments** dropdown lists the recorded dashboards.

To republish an existing timestamped report:

```bash
bash deploy/local/lab.sh routing-dashboard .local/routing-<run>/report.json
```

Legacy reports without request timestamps cannot be plotted honestly against
wall-clock time. Their `pod-final.json` records only the overall container
window; rerun the workload to obtain an exact request timeline. If Grafana
publication fails, the command exits nonzero but retains the completed report
for republication.

Each case starts fresh, genuine Dynamo processes: two SGLang prefill workers,
one SGLang decode worker and the frontend/KV router. There is no Planner or
background user traffic in the experiment, and its metrics are not scraped
into the main dashboard. Default order is metadata **on/off, then off/on**.

The default cache is 96 blocks of 16 tokens, and each query contains a
64-block family prefix plus one unique question block. One family fits, two
do not. Twenty-four serial queries per case use exact token IDs supplied
through `nvext.token_data`, fixed output length and temperature zero. The
report includes corpus, per-query input and output hashes; A/B outputs must
match. Defaults add 250 ms per 1,000 uncached tokens to make missed reuse
visible without pretending to benchmark a GPU.

Only initial seeds and eviction-pressure requests are pinned to workers.
Measured replay requests are **unpinned**. With metadata enabled, every
replay must choose the seeded cache owner and reuse exactly 1,024 of 1,040
input tokens. After forced eviction and reseeding elsewhere, an unpinned
request must follow the new owner. Bounded polling of successful prefill
store events ensures Dynamo has applied prior cache changes before the next
request. The HTTP query-only preview checks token identity and frontend
readiness, not prefill ownership: in disaggregated mode it selects decode.

The control omits worker `--kv-events-config` but keeps
`--router-mode kv --router-temperature 0 --router-kv-events` in both cases.
Using `--no-router-kv-events` would enable approximate route-based prediction
and would not measure the absence of cache metadata. The control must report
zero predicted hits, though the fake engine can still experience chance hits.

Every measured request is checked against per-worker observed counters,
configured capacity and exact modeled generation time. The report compares
token-weighted hit ratio, modeled K/V seconds and measured TTFT p50/p95 and
mean latency. Positive latency-change percentages mean a **regression**;
negative values mean improvement. `seeded_owner_fraction` means affinity to
the original seed, not optimal-routing accuracy after baseline cache churn.
Queries and ownership assertions are deterministic; worker IDs, equal-cost
ties and wall-clock timings are not. This is a serial locality experiment,
not an overloaded-worker routing benchmark.

### Metrics, traces and logs

The dashboard contains a compact inference/Planner overview and collapsed,
type-aware sections covering all metric families discovered from Prometheus.
Sources include Dynamo, its operator, NATS, etcd, the Collector, Grafana,
Tempo, Loki and Prometheus itself. Open a section to display its metrics;
collapsed sections do not issue thousands of queries in the background.

- **Source / Instance** filter the entire dashboard.
- **Counters** show cumulative values and reset-aware per-second rates on
  separate axes. **Gauges** show raw values.
- **Histograms** show p50/p95/p99 and mean, preserving series labels.
- **Raw metric explorer** exposes every live metric name, including histogram
  buckets/counts/sums and metrics registered after the catalog was generated.
- **No data** means there are no samples or the corresponding feature is
  inactive, not a fabricated zero.

Refresh the persisted catalog when new metric families appear:

```bash
bash deploy/local/lab.sh dashboard
```

The same dashboard UID is retained. Grafana picks up the ConfigMap update
automatically; reload the dashboard after the update propagates.
Grafana Explore also has provisioned Tempo and Loki data sources.

#### Observed prefix-cache hit ratio

With `--enable-metrics` (enabled in the local lab), the fake engine exports
observations through the unmodified Dynamo SGLang metrics callback. These
measure actual reuse of the **simulated local prefix cache**, not router
predictions or physical GPU measurements.

The overview includes token-weighted fleet/per-worker ratios, queried/reused
token rates, and a per-request mean alongside the router's predicted mean.
Prometheus metric names (the Collector preserves SGLang's colon prefix) are:

| Metric | Meaning |
|---|---|
| `sglang:simulated_prefix_cache_query_tokens_total` | All prompt tokens at initial prefill admission |
| `sglang:simulated_prefix_cache_hit_tokens_total` | Prompt tokens actually reused from retained local prefix blocks |
| `sglang:simulated_prefix_cache_hit_ratio` | Per-admission hit fraction histogram (`_sum`, `_count`, `_bucket`) |

For a token-weighted ratio per worker:

```promql
sum by (pod, dp_rank) (rate(sglang:simulated_prefix_cache_hit_tokens_total[5m]))
/
sum by (pod, dp_rank) (rate(sglang:simulated_prefix_cache_query_tokens_total[5m]))
```

The ratio is 0-1; Grafana displays it as a percentage. Counters reset on
worker restart. An idle window or decode worker with no local prefix
lookups has no defined ratio, not a zero hit rate.

Observations are recorded once, at the first successful prefill progress.
Chunking, admission retries and preemption re-prefills do not double-count.
The denominator includes partial-block prompt tokens; only complete cached
blocks count as hits. Decode KV received through disaggregation is excluded
from both numerator and denominator. Requests aborted before admission are
not observed; aborts after admission retain their observation.
No-peer calls (`FAKE_BOOTSTRAP_HOST`), including Dynamo's disaggregated health
probes and startup warmup, are excluded without changing their execution.

Raw worker endpoints use the `sglang:simulated_prefix_cache_` prefix and
`dp_rank` label; Dynamo adds worker/model identity and the Collector adds
pod/role labels. The per-request histogram mean is not token-weighted.
Router histograms also count a different population (potentially multiple
routing observations per frontend request), so the comparison is not an
exact request-by-request accuracy measurement.

The separate **Backend-reported cached fraction** panel includes transferred
KV in this mock and must not be interpreted as cross-request prefix reuse.

### Rebuild and stop

```bash
bash deploy/local/lab.sh rebuild  # test a new image, load it, then roll workers
bash deploy/local/lab.sh deploy   # reapply worker config using the recorded image
bash deploy/local/lab.sh stop     # stop only this Colima VM; preserve its data
```

The kubeconfig is `.local/kubeconfig`; use it explicitly for manual commands:

```bash
KUBECONFIG="$PWD/.local/kubeconfig" kubectl -n dynamo-sim get dgd,pods
```

Prometheus retains 48 hours of metrics (up to 3 GB); Tempo retains 24 hours
of traces. Persistent volumes survive `stop`. The lab does not provide
physical GPU utilization, CUDA/NCCL/RDMA execution, real KV tensors, Grove,
Kai scheduling, snapshots, GPU power management or multi-tier KV offload.
Simulated timings are not calibrated GPU benchmarks.

## Native Python setup (alternative)

Clone this repo next to a Dynamo checkout — the bootstrap script expects
`../dynamo` and creates the shared virtualenv at `../.venv`:

```bash
git clone https://github.com/ai-dynamo/dynamo.git
git clone https://github.com/karannparikh/dynamo-sglang-lab.git fake-sglang
```

Then one command, ~5 minutes on a clean machine:

```bash
cd fake-sglang
./scripts/bootstrap_macos.sh
```

It installs cmake/protobuf, Rust, `uv`, creates `../.venv` (Python 3.10),
builds Dynamo's PyO3 bindings from source (there is no macOS wheel), installs
`ai-dynamo` and this package, and verifies the result.

**Safe to re-run.** Every step checks before it acts, so a second run reuses
the existing toolchain and virtualenv and reinstalls nothing already
satisfied (~19s, almost all of it the incremental Rust rebuild). The binding
build is repeated on purpose — it is the only way a change under `dynamo/lib`
reaches the wheel, and cargo's cache keeps it at ~15s rather than ~2.5min.

| Override | Effect |
|---|---|
| `SKIP_BINDINGS=1` | Skip the Rust rebuild when `dynamo._core` already imports (~2s total) |
| `FORCE_RECREATE=1` | Delete and rebuild the virtualenv from scratch |
| `PY_VERSION=3.12` | Use a different Python (see the `asyncio.timeout` note under *Known gaps*) |

If `../.venv` exists but is a different Python than expected, the script stops
with an explicit message rather than half-provisioning against it.

Model metadata ships in the repo: `models/qwen3-0.6b/` holds Qwen3-0.6B's
**config + tokenizer only, no weights** (~15 MB instead of ~1.5 GB), pinned to
a Hugging Face revision so every checkout is byte-identical (see
`models/qwen3-0.6b/SOURCE.md`). The launch scripts use it by default. Pointing
`--model-path` at a local directory also makes Dynamo skip its own model
download entirely, so runs are offline and instant.

To use a different model, or to move to a newer Qwen3 revision, fetch its
metadata and point `MODEL` at it:

```bash
source ../.venv/bin/activate
python scripts/fetch_model_metadata.py <hf-repo-id> models/<name> [REVISION]
MODEL=$PWD/models/<name> ./launch/agg.sh
```

`fetch_model_metadata.py` needs `huggingface_hub` and network access. When
bumping the Qwen3 revision, update `QWEN3_0_6B_REVISION` in the script and
`SOURCE.md` alongside the regenerated files.

## Run native processes

No etcd, no NATS, no Docker — the launch scripts use Dynamo's file-based
discovery backend (`DYN_DISCOVERY_BACKEND=file`) and a TCP request plane.

```bash
./launch/agg.sh            # frontend :8000 + one aggregated worker
./launch/disagg.sh         # frontend :8000 + prefill worker + decode worker
./launch/agg_kv_router.sh  # frontend :8000 + 2 workers behind the KV-aware router
```

```bash
curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"fake-model","messages":[{"role":"user","content":"hello"}],"max_tokens":16}'
```

Any flag is forwarded to the workers, so you can dial the simulation:

```bash
./launch/disagg.sh --fake-kv-bandwidth-gb-s 1 --fake-itl-ms 25
```

## Simulation knobs

| Flag | Default | Effect |
|---|---|---|
| `--fake-prefill-ms-per-1k-tokens` | 55.0 | Prefill cost per 1k recomputed tokens |
| `--fake-prefill-overhead-ms` | 4.0 | Fixed per-prefill-step cost |
| `--fake-kv-generation-ms-per-1k-tokens` | 0.0 | Additional K/V-generation cost for uncached prefill tokens |
| `--fake-itl-ms` | 9.0 | Base inter-token latency |
| `--fake-itl-ms-per-running-req` | 0.35 | ITL added per extra request in the batch |
| `--fake-kv-bandwidth-gb-s` | 64.0 | Simulated KV-transfer bandwidth |
| `--fake-kv-transfer-overhead-ms` | 1.0 | Fixed per-transfer cost |
| `--fake-speedup-ratio` | 1.0 | Divides every simulated sleep (time compression) |
| `--fake-num-kv-blocks` | 8192 | KV pool size; lower it to reach preemption |
| `--fake-bootstrap-timeout-s` | 30.0 | How long decode waits for prefill's KV |

Capacity is `fake_num_kv_blocks * page_size * dcp_size` token slots.
SGLang's standard `--max-total-tokens`, when supplied, takes precedence over
the block count and must be positive and divisible by `page_size * dcp_size`.
Invalid sizes and prompts larger than the worker's KV/context capacity fail
explicitly instead of waiting forever for impossible admission.

KV event publishing is enabled with SGLang's own flag, not a `--fake-` one:

```bash
--page-size 16 --kv-events-config '{"publisher":"zmq","endpoint":"tcp://*:25557"}'
```

Dynamo derives `use_kv_events` from that same config, so one flag turns on both
the engine's publishing and Dynamo's subscription. Give each worker its own
port.

## How disaggregation flows

Dynamo's Rust `PrefillRouter` orchestrates this; the two workers only do the
KV handoff.

```
frontend --> PrefillRouter --> prefill worker   (max_tokens forced to 1)
                                    |
                    yields disaggregated_params {bootstrap_host, port, room}
                    and publishes the room on its bootstrap HTTP server
                                    |
             PrefillRouter stamps it onto the decode leg as bootstrap_info
                                    v
                              decode worker
                    GET /room/{id}  -> Bootstrapping   (real HTTP wait)
                    sleep(payload)  -> Transferring    (simulated bytes)
                                    -> Success
                    decodes from a "prebuilt" batch, leading with the
                    token prefill actually produced
```

`bootstrap_host == "2.2.2.2"` (SGLang's `FAKE_BOOTSTRAP_HOST`) short-circuits
the whole handshake. Dynamo's health-check canary and prefill warmup both use
it; without that path, warmup would block for its 1800 s timeout.

## KV-aware routing

Each worker publishes `BlockStored` / `BlockRemoved` events as its prefix cache
changes. Dynamo's router subscribes, builds a radix tree per worker, and sends
a request to whichever worker already holds the longest matching prefix.

Unreferenced cached blocks are evicted in LRU **leaf-first** order: a parent
cannot disappear while a reusable descendant remains. This keeps fake
SGLang's retained prefixes consistent with Dynamo's router tree under
pressure. Referenced blocks cannot be evicted.

The subtlety worth knowing: Dynamo treats the published `block_hashes` as
**opaque engine ids** and recomputes its own prefix key from the published
`token_ids` (`lib/kv-router/src/zmq_wire/convert.rs`). So the token payload has
to be exactly right — wrong or misaligned tokens produce hashes that silently
never match the frontend's, and routing degrades to round-robin with no error
anywhere. `tests/test_kv_events.py` asserts the published tokens rehash to what
`compute_block_hash_for_seq` gives for the same prompt.

Measured on this stack with two workers (`./launch/agg_kv_router.sh`), a
~6700-token prompt sent twice:

```
pass   overlap_blocks
cold   0
warm   422          <- full prefix hit, routed back to the worker holding it
```

With `--kv-events-config` omitted the same test gives `0` on both passes, which
is what proves the match comes from the published events rather than the
router's own bookkeeping.

> **Known flakiness.** Occasionally the warm pass still reports
> `overlap_blocks=0` on a freshly started stack. The engine is emitting
> correctly in those runs (verified with a raw ZMQ subscriber), so this is a
> delivery or indexer-timing race on the Dynamo side, not in the fake engine.
> A block is published once and never re-announced, so an event lost in that
> window is not recovered. Restarting the stack clears it.

## Layout

```
src/sglang/       the shim: SGLang-shaped facade, no simulation logic
                    (incl. the KV-event wire format, matched to SGLang's)
src/fakeengine/   the simulator: scheduler, KV pool, timing, bootstrap,
                    transfer, KV-event policy
launch/           agg.sh, disagg.sh, agg_kv_router.sh
scripts/          bootstrap_macos.sh, fetch_model_metadata.py
                    traffic.py (bounded OpenAI streaming traffic)
                    routing_scenario.py (deterministic metadata A/B experiment)
deploy/local/     isolated kind/Colima lab, telemetry and full metric dashboard
models/           pinned model metadata (config + tokenizer, no weights)
tests/            contract tests + simulator behavior + disagg + KV events
```

The split is the point: `src/sglang/` translates, `src/fakeengine/` behaves.
Swapping in real SGLang, or driving the simulator from a different facade, only
touches one side.

## Tests

```bash
pytest tests/
pytest ../dynamo/components/src/dynamo/sglang/tests -m unit
```

`tests/test_shim_contract.py` is the drift guard. It re-derives the contract
from Dynamo's own source — every `sglang.*` import, every `server_args.<field>`
and `tokenizer_manager.<attr>` read — and fails by name when Dynamo starts
using something the shim lacks. It understands Dynamo's `_compat.py` N/N-1
fallback groups, requiring only that *one* alternative in each required group
resolves. Explicit None/no-op fallbacks also permit an import to be absent.

### Known gaps

Three of Dynamo's own SGLang tests still fail, none of them core:

- `test_video_encode_real`, `test_compat_supports_tensor_image_sizes` —
  multimodal image/video paths, deliberately stubbed out.
- `test_cancelled_sync_engine_route_keeps_engine_routes_serialized` — uses
  `asyncio.timeout`, which is Python 3.11+. Unrelated to the shim; the
  bootstrap pins 3.10 because `ai-dynamo`'s `aisimulate` dependency is marked
  `python_version >= '3.11'` and ships only an sdist. Build on 3.12 (Rust is
  already installed by then) if you need that test.

## Not simulated yet

- **Multi-rank KV events.** One DP rank per worker. The per-rank port offset is
  implemented, but nothing drives `dp_size > 1`.
- **Multi-rank FPM.** The simulated FPM publisher supports one DP rank per
  worker; enabling it with `dp_size > 1` raises an explicit error.
- Multimodal, embedding/rerank, diffusion, LoRA, and weight updates — these
  raise a clear `NotImplementedError` rather than pretending.
