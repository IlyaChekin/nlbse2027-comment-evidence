# Follow-up experiments for selective reading

The experiments use the released NLBSE dataset and a single six-block CodeT5+ classifier per run. The public data in the upstream file named `test` are the released validation split. [PLAN.md](PLAN.md) gives the fixed comparisons. [results.md](results.md) and [results.csv](results.csv) summarize the results. The ordered predictions, thresholds, row manifests, and logs are included. The standalone competition release ships the frozen checkpoint. The commands below regenerate the experimental checkpoints, whose original hashes are recorded in the comparison reports.

## Reproduce the experiments

From the extracted root, install the versions in `verification/requirements-resolved.txt`. Training and evaluation use CUDA. Run in a fresh copy when reproducing outputs.

```bash
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
python3 nlbse/fetch_inputs.py --training
gzip -dc nlbse/data/split_manifest.csv.gz > nlbse/data/split_manifest.csv
bash nlbse/portfolio/fetch_backbones.sh

python3 nlbse/round6/train_context.py
python3 nlbse/round6/evaluate_training.py holdout
python3 nlbse/round6/evaluate_training.py public

python3 nlbse/round6/router.py learn
python3 nlbse/round6/router.py public

python3 nlbse/round6/project_model.py
python3 nlbse/round6/masked_context.py public
python3 nlbse/round6/masked_context.py project
python3 nlbse/round6/report.py
```

To rebuild the supplement, run `python3 nlbse/paper/build_context_followup.py` and use IEEEtran to compile `nlbse/paper/context_followup/main.tex`.

## Run inference

Use a CSV with the `Function` column. Both tools preserve row order and emit `vulnerability,MAT`.

```bash
python3 nlbse/round6/router.py predict --input functions.csv --output routed.csv
python3 nlbse/round5/test_context.py predict \
  --config nlbse/round6/training/short_window.json \
  --checkpoint nlbse/round6/training/fixed384 \
  --input functions.csv --output fixed384.csv
```

For the mixed-length checkpoint substitute `nlbse/round6/training/mixed`. Add `--device cpu` after `predict` for CPU inference. The stored 384+64 configuration uses the same model on both passes and the thresholds 0.70 and 0.50.

The base release and primary tool-paper version are available in the earlier context study. Its verification procedure called `t4_context.sh` remains applicable to that frozen candidate. New times in this follow-up are local RTX 4090 measurements.
