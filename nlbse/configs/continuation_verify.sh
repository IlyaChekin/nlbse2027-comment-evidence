#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
for n in $(seq 1 480); do
  if test -s nlbse/runs/own6_packed_public/metrics.json; then break; fi
  sleep 5
done
test -s nlbse/runs/own6_packed_public/metrics.json
.venv/bin/python -u nlbse/round2_report.py >nlbse/reports/round2_verify_scores.log 2>&1
.venv/bin/python -u nlbse/verify_entry.py --device cpu >nlbse/reports/round2_verify_entry.log 2>&1
read -r winner checkpoint < <(.venv/bin/python - <<'PY'
import json
with open("nlbse/reports/round2_selection.json") as f:
    selection = json.load(f)
winner = selection["winner"]
print(winner, "nlbse/" + selection["candidates"][winner]["checkpoint"])
PY
)
exec 9>verification/gpu_queue.lock
flock -w 60 9
mkdir -p nlbse/runs/entry_ten_repeat_smoke
.venv/bin/python -u nlbse/cuda_launch.py nlbse/pipeline.py evaluate --size 256 --run entry_ten_repeat_smoke \
  --bucket-window 2048 --profile-reps 10 --profile-cuda --checkpoint "$checkpoint" \
  --thresholds "nlbse/reports/$(printf '%s_frozen_thresholds.json' "$winner")" \
  >nlbse/runs/entry_ten_repeat_smoke/run.log 2>&1
