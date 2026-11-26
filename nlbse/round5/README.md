# Conditional reading for NLBSE 2027

The artifact contains one frozen six-block CodeT5+ classifier, a policy for reading truncated comments, both joint predictions in input order, and the tool-paper source. The public research comparison is described in [results.md](results.md).

## Run the submitted-tool candidate locally

Work from the extracted archive root with the dependencies in `verification/requirements-resolved.txt`. The input CSV must have a column named `Function`.

```bash
TOKENIZERS_PARALLELISM=false python3 nlbse/round5/test_context.py predict \
  --input functions.csv --output predictions.csv
```

The default configuration is `nlbse/round5/selected_inference.json`. Use `--device cpu` for CPU inference. The output columns are `vulnerability,MAT`. The same checkpoint is used for every window.

## Reproduce the public evaluation

```bash
python3 nlbse/fetch_inputs.py
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
python3 nlbse/round5/test_context.py public --out nlbse/round5/replay/public
python3 nlbse/round5/test_context.py public --diagnostic --repetitions 1 \
  --out nlbse/round5/replay/diagnostic
python3 nlbse/round5/test_context.py public --selection nlbse/round5/budget_selection.json \
  --out nlbse/round5/replay/budget
```

The public parquet file named `test` is the released validation split. The final hidden evaluation is maintained by the organizers. A T4 instance can replay the notebook's ten-repetition profiling protocol for the prior rule and the selected configuration.

```bash
bash nlbse/round5/t4_context.sh
```

## Reproduce the policy selection and paper

Use a fresh extracted copy for work that writes the experiment receipts.

```bash
python3 nlbse/fetch_inputs.py --training
gzip -dc nlbse/data/split_manifest.csv.gz > nlbse/data/split_manifest.csv
python3 nlbse/round5/cache_context.py
python3 nlbse/round5/select_context.py
python3 nlbse/round5/context_budget.py
```

Replay the public evaluation commands above with `replay/` changed to `runs/`. Then run `python3 nlbse/round5/report_context.py` and `python3 nlbse/paper/build_submission.py`. Compile the combined tool paper from `nlbse/paper/submission/main.tex` with IEEEtran. Set the public artifact URL in `nlbse/paper/submission/submission-metadata.tex`.

The provided weights reconstruct the same model used for this study. To train them again on a supported CUDA host, first run the official training download and restore the split above, then run the following with the frozen FP16 checkpoint in a separate copy.

```bash
bash nlbse/portfolio/fetch_backbones.sh
python3 nlbse/portfolio/neural.py codet5p
python3 nlbse/portfolio/neural.py codet5p --full
python3 nlbse/portfolio/t5_entry.py export
```

See [PLAN.md](PLAN.md) for the training holdout and the public selection protocol. The local score estimates and repeated timings use RTX 4090.
