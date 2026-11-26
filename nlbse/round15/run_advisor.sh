#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
nlbse_round=nlbse/round15
nlbse_python=.venv/bin/python

if [[ $# -eq 1 ]]; then
  routing_job=$1
  [[ "$routing_job" =~ ^[0-9]+$ ]] || exit 2
  while true; do
    job_stat=$(ps -p "$routing_job" -o stat= || true)
    if [[ -z "$job_stat" || "$job_stat" == Z* ]]; then
      break
    fi
    sleep 15
  done
fi

[[ -s "$nlbse_round/precision/42_control/test.npz" ]]
{
  "$nlbse_python" -u nlbse/cuda_launch.py "$nlbse_round/precision.py" fit --mode advisor
  for part in dev test; do
    "$nlbse_python" -u nlbse/cuda_launch.py "$nlbse_round/precision.py" score --mode advisor --part "$part"
  done
  "$nlbse_python" -u "$nlbse_round/precision.py" tune --study advisor
  "$nlbse_python" -u "$nlbse_round/precision.py" test --study advisor
} > "$nlbse_round/precision_advisor.log" 2>&1
