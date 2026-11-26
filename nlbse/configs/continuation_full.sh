#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
exec 9>verification/gpu_queue.lock
flock -w 300 9
run() {
  local name="$1"
  shift
  mkdir -p "nlbse/runs/$name"
  if test -s "nlbse/runs/$name/metrics.json"; then
    if [[ "$1" != train ]] || test -s "nlbse/runs/$name/checkpoint/model.safetensors"; then return; fi
  fi
  .venv/bin/python -u nlbse/cuda_launch.py nlbse/pipeline.py "$@" --run "$name" >"nlbse/runs/$name/run.log" 2>&1
}
teacher=nlbse/runs/final_fit/checkpoint
choice="$(.venv/bin/python nlbse/continuation_select.py pilot)"
teacher_args=()
if [[ "$choice" == kd ]]; then teacher_args=(--teacher-from "$teacher"); fi
run compact6_full train --weighted --train-attention sdpa --bucket-window 2048 \
  --init-from "$teacher" --retain-layers 1 3 5 7 9 11 "${teacher_args[@]}"
run own12_packed_tune evaluate --internal --bucket-window 2048 --profile-reps 1 \
  --checkpoint "$teacher" --thresholds nlbse/runs/final_fit/thresholds.json
run own6_packed_tune evaluate --internal --bucket-window 2048 --profile-reps 1 \
  --checkpoint nlbse/runs/compact6_full/checkpoint --thresholds nlbse/runs/compact6_full/thresholds.json
winner="$(.venv/bin/python nlbse/continuation_select.py freeze)"
printf '%s Frozen candidate %s\n' "$(date -Iseconds)" "$winner"
run official_packed_public evaluate --bucket-window 2048 --profile-reps 1
run own12_packed_public evaluate --bucket-window 2048 --profile-reps 1 --checkpoint "$teacher" \
  --thresholds nlbse/reports/own12_frozen_thresholds.json
run own6_packed_public evaluate --bucket-window 2048 --profile-reps 1 \
  --checkpoint nlbse/runs/compact6_full/checkpoint --thresholds nlbse/reports/own6_frozen_thresholds.json
