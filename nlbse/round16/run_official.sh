#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
nlbse_round=nlbse/round16
[[ -s "$nlbse_round/projects/42/results.json" && -s "$nlbse_round/projects/101/results.json" ]]

for phase in fit tune public; do
  .venv/bin/python -u nlbse/cuda_launch.py "$nlbse_round/local_public.py" "$phase" > "$nlbse_round/official_$phase.log" 2>&1
done
