#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
nlbse_round=nlbse/round15

for fold in "$@"; do
  case "$fold" in 0|1|2|3|4) ;; *) exit 2 ;; esac
  printf 'Starting fold %s\n' "$fold"
  .venv/bin/python -u nlbse/cuda_launch.py "$nlbse_round/crossfit.py" pilot --fold "$fold" > "$nlbse_round/fold_${fold}_train.log" 2>&1
  .venv/bin/python -u "$nlbse_round/audit_fold.py" --fold "$fold" > "$nlbse_round/audit_${fold}.log" 2>&1
  printf 'Verified fold %s\n' "$fold"
done
