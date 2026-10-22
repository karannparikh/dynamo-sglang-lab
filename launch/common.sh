# Shared helpers for the fake-sglang launch scripts.
# Mirrors dynamo/examples/common/launch_utils.sh: one terminal, every process
# backgrounded, and the whole group torn down when any member exits.

# This checkout, wherever it was cloned; REPOS is the directory holding it and
# its siblings (dynamo, .venv).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPOS="${REPOS:-$(dirname "$ROOT")}"
VENV="${VENV:-$REPOS/.venv}"
PY="$VENV/bin/python"

# No etcd and no NATS: DYN_DISCOVERY_BACKEND=file plus a TCP request plane is a
# fully local control plane (dynamo lib/runtime/src/distributed.rs).
export DYN_DISCOVERY_BACKEND="${DYN_DISCOVERY_BACKEND:-file}"
export DYN_LOG="${DYN_LOG:-info}"

MODEL="${MODEL:-$ROOT/models/qwen3-0.6b}"

require_model() {
  # A local directory must exist. A bare HF repo id (no leading slash or dot)
  # is passed through -- Dynamo downloads the metadata itself in that case.
  case "$MODEL" in
    /*|./*|../*)
      [ -d "$MODEL" ] && return 0
      echo "error: MODEL=$MODEL does not exist." >&2
      echo "Fetch metadata (config + tokenizer, no weights) with:" >&2
      echo "  $PY $ROOT/scripts/fetch_model_metadata.py Qwen/Qwen3-0.6B $MODEL" >&2
      exit 1
      ;;
  esac
}

wait_any_exit() {
  # Return as soon as ANY child exits, so a crashed worker ends the run
  # instead of leaving a half-up deployment that looks healthy.
  local pid
  while true; do
    for pid in "$@"; do
      kill -0 "$pid" 2>/dev/null || return 0
    done
    sleep 1
  done
}

banner() {
  echo
  echo "=============================================================="
  echo " $*"
  echo " SIMULATED ENGINE -- no weights, no real inference."
  echo "=============================================================="
  echo
}
