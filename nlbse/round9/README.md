# CUDA fusion, grouped suffix batches and a calibrated early exit

Every run uses the same CodeT5+ encoder as the preceding release, official NLBSE data and the same 30,655 public examples. The first four blocks produce reusable hidden states. Later attention still sees all the tokens. Calibration and the student use official training groups only.

## Recommended operating points

| Mode | Mean F1 | Wall time | Estimated GFLOPs | Local formula |
|---|---:|---:|---:|---:|
| Prior release, selective token updates | 0.9311 | 15.33 s | 485,224 | 0.9234 |
| Fused kernel, exact ordered decisions | 0.9311 | 10.87 s | 485,224 | 0.9244 |
| Length-based fused batching | 0.9310 | 9.93 s | 485,216 | 0.9246 |
| Asymmetric early exit, grouped and length-batched | 0.9305 | 9.55 s | 421,054 | 0.9286 |
| Aggressive early exit, grouped and length-batched | 0.9272 | 9.19 s | 399,788 | 0.9282 |
| Higher-quality 65% token updates with fusion | 0.9330 | 11.19 s | 509,236 | 0.9239 |

The recommended `balanced` mode saves 37.7% end-to-end time and 13.2% estimated operations against the prior release. It exits after four blocks for 23,121 functions, and finishes selective updates on 7,534. Its paired mean-F1 interval against the release is −0.26 to +0.13 percentage points. The measurements are local on an RTX 4090. The score applies the official formula to measured forward time and analytic padded GFLOPs. Organizer T4 and hidden evaluations are separate. See [unrounded table](results.csv) and [final interleaved benchmark](combined/analysis.json).

## What changed and what we tested

- The [memory-efficient PyTorch attention kernel](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html) uses the same T5 relative bias, padding mask and score scaling. The 50% variant keeps all 61,310 decisions identical to the eager reference. The 65% variant changes one. [50% trials](fused/analysis.json) · [65% trials](fused_quality/analysis.json).
- On 193,100 train examples we fit equal-size early readouts on frozen features, testing no dropout/weight decay, dropout 0.18 plus weight decay 0.025, and the same regularization with temperature-2 distillation. Independent internal review mean F1 is 0.9093, 0.9004 and 0.9084 respectively. The calibration fold ties the unregularized and distilled readouts at 0.9101. The fixed tie break chooses the unregularized one. The held-out regression combination of early and late logits also gives no improvement over the matched late readouts. [Student fits](frozen/policy.json) · [Readout combination](readout_fusion.json).
- The asymmetric rule sends examples to the last two blocks when the first vulnerability probability lies strictly between 0.025 and 0.86. The cutoffs and late mode were picked on the 4,292 calibration rows. The 4,294-row exact-text-disjoint review achieved 0.9288. The alternative learned gate preserved public F1 in its higher-quality setting and was slower. [Grouped folds and gates](frozen/asymmetric.json) · [Per-small-batch routing](public/analysis.json).
- Routing inside each small batch initially ran in 12.64 seconds despite using fewer operations. Moving the readout/gate to the GPU and pooling uncertain rows across each 2,048-function work block reduced the corresponding aggressive-mode time to 9.77 seconds. The four-block states stay resident and the late stage does no prefix recomputation. [Grouped suffix passes](packed/analysis.json).
- Raising every batch to 96 increased padded compute on the full public set. The train-selected variable-size alternative uses an 8,192 padded-token limit and a quadratic attention cap. Applied alone it changes two binary decisions and reduces peak memory from about 2.43 GiB to 1.32 GiB. Combined with the early exit it improves real time compared with the same rule and fixed batches. For the balanced mode it changes one decision and slightly reduces padded work. [Fixed batch sweep](batch/analysis.json) · [Length sweep](length_batches/analysis.json).

The paired comparisons use project and file groups for bootstrap intervals. The encoder and token-policy revisions are recorded in the [feature cache manifest](cache/manifest.json). The distributed artifact includes fit summaries, heads, configurations and ordered prediction receipts. The frozen feature arrays can be regenerated from the official data.

## Reproduce

Use the PyTorch/CUDA environment and official-data preparation from the baseline release. Run timed stages independently.

```bash
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
python3 nlbse/fetch_inputs.py --training
gzip -dc nlbse/data/split_manifest.csv.gz > nlbse/data/split_manifest.csv
python3 nlbse/round9/fused_attention.py
python3 nlbse/round9/cache_hybrid.py
python3 nlbse/round9/tune_hybrid.py
python3 nlbse/round9/hybrid.py benchmark
python3 nlbse/round9/batches.py
python3 nlbse/round9/packed_suffix.py benchmark
python3 nlbse/round9/fuse_readouts.py
python3 nlbse/round9/thresholds.py
python3 nlbse/round9/packed_suffix.py benchmark --rules nlbse/round9/frozen/asymmetric.json \
    --output nlbse/round9/asymmetric --repeat 2
python3 nlbse/round9/length_batches.py
python3 nlbse/round9/fused_attention.py --mode query_even_65 \
    --output nlbse/round9/fused_quality --repeat 2
python3 nlbse/round9/combined_schedule.py
python3 nlbse/round9/report.py
```

Production input has a `Function` column. Output columns are `vulnerability,MAT` in the same row order. Selection of the modes is encoded in [the release registry](release_policy.json).

```bash
python3 nlbse/round9/predict.py --mode balanced --input functions.csv --output predictions.csv
```
