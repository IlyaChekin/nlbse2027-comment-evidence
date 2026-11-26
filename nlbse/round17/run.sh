#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
round=nlbse/round17
python=.venv/bin/python

gpu() {
  local script=$1
  shift
  "$python" -u nlbse/cuda_launch.py "$round/$script" "$@"
}

case "${1:-}" in
  project)
    for seed in 42 101; do
      gpu view_train.py projects --seed "$seed" > "$round/train_projects_$seed.log" 2>&1
      gpu view_eval.py project --seed "$seed" > "$round/eval_projects_$seed.log" 2>&1
    done
    ;;
  official)
    decision=$("$python" "$round/view_eval.py" plan)
    read -r -a modes <<< "$decision"
    gpu view_train.py official --modes "${modes[@]}" > "$round/train_official.log" 2>&1
    for phase in tune public; do
      gpu view_eval.py "$phase" > "$round/eval_official_$phase.log" 2>&1
    done
    ;;
  router)
    for phase in project tune public oracle; do
      gpu view_route.py "$phase" > "$round/router_$phase.log" 2>&1
    done
    ;;
  representation)
    gpu representation.py > "$round/teacher_representation.log" 2>&1
    for seed in 42 101; do
      gpu view_train.py projects --seed "$seed" --representation > "$round/train_representation_$seed.log" 2>&1
      gpu view_eval.py project --seed "$seed" --representation > "$round/eval_representation_$seed.log" 2>&1
    done
    ;;
  latent)
    gpu view_route.py oracle > "$round/router_oracle_verified.log" 2>&1
    gpu latent_probe.py > "$round/latent_probe.log" 2>&1
    ;;
  *)
    exit 2
    ;;
esac
