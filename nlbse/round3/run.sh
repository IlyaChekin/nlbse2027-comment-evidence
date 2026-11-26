#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
.venv/bin/python -u nlbse/round3/simple.py
exec 9>verification/gpu_queue.lock
flock -w 60 9
.venv/bin/python -u nlbse/cuda_launch.py nlbse/round3/ideas.py
.venv/bin/python -u nlbse/cuda_launch.py nlbse/round3/early_followup.py
flock -u 9
.venv/bin/python -u nlbse/round3/report.py
