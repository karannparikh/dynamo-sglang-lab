#!/usr/bin/env bash
# Aggregated: one fake worker does prefill and decode.
#   ./launch/agg.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_model
banner "fake-sglang aggregated: frontend :8000 + 1 worker"

trap 'echo; echo "Cleaning up..."; kill 0' EXIT

"$PY" -m dynamo.frontend --http-port 8000 &
FRONTEND=$!

DYN_SYSTEM_PORT=8081 "$PY" -m dynamo.sglang \
  --model-path "$MODEL" \
  --served-model-name "${SERVED_NAME:-fake-model}" \
  --skip-tokenizer-init \
  "$@" &
WORKER=$!

wait_any_exit "$FRONTEND" "$WORKER"
