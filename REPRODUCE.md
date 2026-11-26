# Reproduction

## Verify the published predictions

Use Python 3.11 or 3.12, a CUDA device, CUDA-enabled PyTorch 2.8, and Node.js 22. The model ZIP is a separate release asset named `NLBSE2027_round38_paired_rule_tool.zip`. Download it and its `.sha256` into the root of this source tree. Set `NLBSE_NODE` to the absolute path of the Node executable if it is outside PATH. The pinned inference requirements are provided in the source tree and the model archive.

```bash
sha256sum -c NLBSE2027_round38_paired_rule_tool.sha256
python -m pip install -r requirements-inference.txt
python nlbse/fetch_inputs.py
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 RAYON_NUM_THREADS=4 NLBSE_SCANNER_WORKERS=6
python nlbse/round12/verify_release.py \
  NLBSE2027_round38_paired_rule_tool.zip \
  nlbse/data/raw/test-00000-of-00001.parquet \
  verification/public --entry nlbse/round25/predict_sections.py --cuda-launch
```

The verifier extracts the model to a fresh directory, uses only `Function` for inference, and compares all 61,310 ordered binary decisions. Read `verification/public/receipt.json` for hashes and `verification/public/cli.log` for logs. [Verified release receipt](evidence/clean_release/receipt.json) · [Recorded quality and timing](evidence/round38-metrics.json)

For ordinary inference, extract the model ZIP and run from its root:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NLBSE_SCANNER_WORKERS=6 \
  python nlbse/round25/predict_sections.py --input functions.csv --output predictions.csv
```

To verify with Colab T4, select a T4 runtime, upload or fetch the two ZIPs, install compatible PyTorch and Node versions, and run the same public verifier after extracting the source ZIP. Local model times depend on the GPU and CPU. Final ranking uses the organizers' notebook and hidden split.

## Refit the final paired correction

This CPU-only check reruns the same objective on the 42,750 numerical training rows and 3,870 within-file pairs. The inputs include only numerical features, row identifiers, and labels. It checks coefficients and the recalibrated cutoff within an absolute tolerance of 0.00002 across numerical libraries, and requires identical calibration and review decisions at that cutoff. It uses the same `fit`, `predict`, and cutoff functions as the training code.

```bash
python -m pip install numpy==2.2.6 scipy==1.18.1 scikit-learn==1.7.2
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python reproduction/refit_round38.py
```

## Training, tuning, and additional experiments

Fetch pinned training data and models with `python nlbse/fetch_inputs.py --training`, `bash nlbse/portfolio/fetch_backbones.sh`, and the remaining per-model fetch scripts referenced by the stage READMEs. The full historical recipes are staged. The model archive and the exact correction replay above provide two starting points for verification without rebuilding the historical checkpoints. The annotated stage READMEs give the source paths, fold settings, and commands for producing upstream feature caches and checkpoints. Data and large intermediate checkpoints are regenerated or downloaded as described there.

1. Run the [file-disjoint compact distillation recipe](nlbse/round15/README.md) and the [full-context compact encoder](nlbse/round16/README.md).
2. Train [the conditional rule estimator](nlbse/round20/README.md) and reproduce [short-circuit rule outputs](nlbse/round22/README.md). Build the [sparse token and section features](nlbse/round24/README.md) and [canonical MAT rules](nlbse/round26/README.md).
3. Use the [conditional score corrections](nlbse/round28/README.md), [first-rule calibration](nlbse/round32/README.md), [pooled projection](nlbse/round34/README.md), [deferred region correction](nlbse/round37/README.md), and [final paired correction](nlbse/round38/README.md). Stage scripts that refer to an earlier tool ZIP expect it at the path created by that stage's build script.
4. Verify the [matched 64-token comparison](nlbse/round56/README.md) from the recorded predictions and timing, or rerun after generating the Round 55 span checkpoint and the release's research-stage inputs. Reconstruct archived context using the upstream fetch scripts and pinned identifiers. Only the public inputs are needed for the frozen release verification.

All fits and threshold searches use training and its internal and project holdouts. The public split is used for evaluation. The review reports note the remaining project/file overlap in the official training split. The archive and verification receipt identify the exact public release independently of the comparison scripts.
