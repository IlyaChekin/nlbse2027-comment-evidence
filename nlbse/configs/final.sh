#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export TOKENIZERS_PARALLELISM=false
PYTHON=.venv/bin/python
run() {
  local name="$1"
  shift
  mkdir -p "nlbse/runs/$name"
  if test -s "nlbse/runs/$name/metrics.json"; then
    if [[ "$1" != train ]] || test -s "nlbse/runs/$name/checkpoint/model.safetensors"; then return; fi
  fi
  "$PYTHON" -u nlbse/pipeline.py "$@" --run "$name" >"nlbse/runs/$name/run.log" 2>&1
}
read -r selected length < <("$PYTHON" nlbse/report.py select)
args=(--length "$length")
if [[ "$selected" == "pilot_weighted_512" ]]; then args+=(--weighted); fi
run final_fit train "${args[@]}"
run official_public evaluate --fixed --profile-reps 1
run final_public evaluate --length "$length" --profile-reps 1 --checkpoint nlbse/runs/final_fit/checkpoint --thresholds nlbse/runs/final_fit/thresholds.json
"$PYTHON" nlbse/report.py summarize
