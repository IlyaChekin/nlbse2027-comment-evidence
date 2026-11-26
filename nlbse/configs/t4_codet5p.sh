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
python3 nlbse/portfolio/t5_entry.py compare --public --profile-reps 10 --out nlbse/portfolio/runs/t4_codet5p
