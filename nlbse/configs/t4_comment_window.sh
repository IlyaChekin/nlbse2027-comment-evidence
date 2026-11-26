#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
python3 - <<'PY'
import torch

device = torch.cuda.get_device_name()
if "T4" not in device:
    raise SystemExit(f"The detected CUDA device is {device}. This replay is for the Colab T4 profile.")
PY
python3 nlbse/fetch_inputs.py
python3 nlbse/round4/entry.py evaluate --head-only --profile-reps 10 --profile-cuda --out nlbse/round4/t4_head
python3 nlbse/round4/entry.py evaluate --profile-reps 10 --profile-cuda --out nlbse/round4/t4_comment
python3 nlbse/score_runtime.py nlbse/round4/{t4_head,t4_comment}/metrics.json \
    | tee nlbse/round4/t4_formula.json
