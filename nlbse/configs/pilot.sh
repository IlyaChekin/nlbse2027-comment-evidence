#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export TOKENIZERS_PARALLELISM=false
PYTHON=.venv/bin/python
run() {
  local name="$1"
  shift
  mkdir -p "nlbse/runs/$name"
  if test -s "nlbse/runs/$name/metrics.json"; then return; fi
  "$PYTHON" -u nlbse/pipeline.py "$@" --run "$name" >"nlbse/runs/$name/run.log" 2>&1
}
"$PYTHON" -u nlbse/pipeline.py prepare >nlbse/reports/data_audit.log 2>&1
run official_smoke evaluate --size 256 --fixed --profile-reps 2
run pilot_bce_512 train --pilot
run pilot_weighted_512 train --pilot --weighted
run pilot_bce_256 train --pilot --length 256
