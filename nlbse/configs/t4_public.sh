#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
python3 - <<'PY'
import torch

device = torch.cuda.get_device_name()
if "T4" not in device:
    raise SystemExit(f"This is the official-hardware run. The detected CUDA device is {device}.")
PY

python3 nlbse/fetch_inputs.py
python3 nlbse/pipeline.py evaluate --run t4_baseline_fixed --fixed --profile-reps 10 --profile-cuda
python3 nlbse/pipeline.py evaluate --run t4_entry --bucket-window 2048 --profile-reps 10 --profile-cuda \
    --checkpoint nlbse/submission/entry_checkpoint --thresholds nlbse/submission/entry_thresholds.json
python3 nlbse/score_runtime.py nlbse/runs/{t4_baseline_fixed,t4_entry}/metrics.json \
    | tee nlbse/reports/t4_public_formula.json
