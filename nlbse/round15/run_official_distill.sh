#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
nlbse_round=nlbse/round15
nlbse_python=.venv/bin/python

if [[ $# -eq 1 ]]; then
  advisor_job=$1
  [[ "$advisor_job" =~ ^[0-9]+$ ]] || exit 2
  while true; do
    job_stat=$(ps -p "$advisor_job" -o stat= || true)
    if [[ -z "$job_stat" || "$job_stat" == Z* ]]; then
      break
    fi
    sleep 15
  done
fi

for stage in fit tune public fixed_fp; do
  "$nlbse_python" -u nlbse/cuda_launch.py "$nlbse_round/official_distill.py" "$stage" > "$nlbse_round/official_distill_${stage}.log" 2>&1
done
