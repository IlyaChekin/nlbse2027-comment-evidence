#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
nlbse_round=nlbse/round15
nlbse_python=.venv/bin/python

if [[ $# -eq 1 ]]; then
  precision_job=$1
  [[ "$precision_job" =~ ^[0-9]+$ ]] || exit 2
  while true; do
    job_stat=$(ps -p "$precision_job" -o stat= || true)
    if [[ -z "$job_stat" || "$job_stat" == Z* ]]; then
      break
    fi
    sleep 15
  done
fi

"$nlbse_python" -u nlbse/cuda_launch.py "$nlbse_round/routing_bench.py" tune > "$nlbse_round/routed_tune.log" 2>&1
"$nlbse_python" -u nlbse/cuda_launch.py "$nlbse_round/routing_bench.py" public > "$nlbse_round/routed_public.log" 2>&1
