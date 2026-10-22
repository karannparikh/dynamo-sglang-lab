#!/usr/bin/env bash
# Disaggregated: a prefill worker hands KV to a decode worker.
#
# Dynamo's Rust PrefillRouter orchestrates this: prefill yields
# {bootstrap_host, bootstrap_port, bootstrap_room} as disaggregated_params, the
# router stamps it onto the decode leg as bootstrap_info, and the two workers
# rendezvous over the prefill worker's bootstrap HTTP server.
#
#   ./launch/disagg.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_model

BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-18998}"
banner "fake-sglang disaggregated: frontend :8000 + prefill + decode (bootstrap :$BOOTSTRAP_PORT)"

trap 'echo; echo "Cleaning up..."; kill 0' EXIT

"$PY" -m dynamo.frontend --http-port 8000 &
FRONTEND=$!

DYN_SYSTEM_PORT=8081 "$PY" -m dynamo.sglang \
  --model-path "$MODEL" \
  --served-model-name "${SERVED_NAME:-fake-model}" \
  --skip-tokenizer-init \
  --disaggregation-mode prefill \
  --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
  "$@" &
PREFILL=$!

DYN_SYSTEM_PORT=8082 "$PY" -m dynamo.sglang \
  --model-path "$MODEL" \
  --served-model-name "${SERVED_NAME:-fake-model}" \
  --skip-tokenizer-init \
  --disaggregation-mode decode \
  "$@" &
DECODE=$!

wait_any_exit "$FRONTEND" "$PREFILL" "$DECODE"
