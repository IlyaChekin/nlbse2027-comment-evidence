#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
python3 - <<'PY'
import torch

device = torch.cuda.get_device_name()
if "T4" not in device:
    raise SystemExit(f"Detected {device}. The public profiling replay uses a Colab T4.")
PY
python3 nlbse/fetch_inputs.py
read -r choice selection < <(python3 - <<'PY'
import json
from pathlib import Path
root = Path("nlbse/round5")
mode = json.loads((root/"selected_inference.json").read_text())["config"]["name"]
for candidate in ("selection.json", "budget_selection.json"):
    if mode in json.loads((root/candidate).read_text())["candidates"]:
        print(mode, root/candidate)
        break
PY
)
python3 nlbse/round5/test_context.py public --profile-reps 10 --repetitions 1 \
  --mode reference --out nlbse/round5/t4
if [[ "$choice" != reference ]]; then
  python3 nlbse/round5/test_context.py public --profile-reps 10 --repetitions 1 \
    --selection "$selection" --mode "$choice" --out nlbse/round5/t4
fi
