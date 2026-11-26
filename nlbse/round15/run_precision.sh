#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
nlbse_round=nlbse/round15
nlbse_python=.venv/bin/python

if [[ $# -eq 1 ]]; then
  fold_queue_pid=$1
  [[ "$fold_queue_pid" =~ ^[0-9]+$ ]] || exit 2
  while ! [[ -s "$nlbse_round/fold_0/audit.json" && -s "$nlbse_round/fold_1/audit.json" &&
             -s "$nlbse_round/fold_2/audit.json" && -s "$nlbse_round/fold_3/audit.json" &&
             -s "$nlbse_round/fold_4/audit.json" ]]; do
    fold_queue_stat=$(ps -p "$fold_queue_pid" -o stat= || true)
    if [[ -z "$fold_queue_stat" || "$fold_queue_stat" == Z* ]]; then
      printf 'Out-of-fold training stopped before all five audits were written\n' >&2
      exit 1
    fi
    sleep 15
  done
fi

"$nlbse_python" -u "$nlbse_round/mine.py" > "$nlbse_round/mining.log" 2>&1
(
  "$nlbse_python" -u "$nlbse_round/precision_gate.py" tune
  "$nlbse_python" -u "$nlbse_round/precision_gate.py" test
) > "$nlbse_round/veto.log" 2>&1 &
veto_pid=$!

for mode in control fp90 joint_errors; do
  (
    "$nlbse_python" -u nlbse/cuda_launch.py "$nlbse_round/precision.py" fit --mode "$mode"
    for part in dev test; do
      "$nlbse_python" -u nlbse/cuda_launch.py "$nlbse_round/precision.py" score --mode "$mode" --part "$part"
    done
  ) > "$nlbse_round/precision_${mode}.log" 2>&1
done
"$nlbse_python" -u "$nlbse_round/precision.py" tune > "$nlbse_round/precision_tune.log" 2>&1
"$nlbse_python" -u "$nlbse_round/precision.py" test > "$nlbse_round/precision_test.log" 2>&1
wait "$veto_pid"
