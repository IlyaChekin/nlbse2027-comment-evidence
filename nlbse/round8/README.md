# Faster inference and more useful context in one frozen CodeT5+

All comparisons use the 30,655 official public functions and the same frozen six-block CodeT5+ weights. Timings are from one RTX 4090. Mean F1 averages the positive-class F1 of vulnerability and MAT. The local score applies the published formula to sampled forward time and estimated padded GFLOPs. The organizers' score requires their hidden data and T4 measurements.

## Exact speedup

The fast tokenizer can defer materializing character offsets. The C-style scanner only requests them for inputs beyond the head limit. A single preprocessing worker prepares one block ahead while the GPU scores the current block. The input blocks, batches, model calls, thresholds, scores, and output order are preserved. The two processes have the same GPU memory peak in the measured run.

Deferred offsets alone reduce public median wall time from 22.593 to 21.915 seconds with all probabilities exactly equal. In the separate three-pair benchmark, deferred offsets and CPU/GPU overlap reduce median wall time from 23.002 to 15.635 seconds, a 32.0% reduction. The three paired savings are 6.75, 7.36, and 7.51 seconds. Measured forward time in isolation rises from 13.542 to 14.423 seconds while CPU preparation overlaps it. The estimated GFLOPs and F1 0.927557 stay identical. [Exact timing receipt](overlap/analysis.json).

## Improving decisions without replacing the encoder

The learned variants use explicit indicators for TODO, FIXME, XXX, and HACK inside C-style comments, plus text length. A trained two-output readout takes these features and outputs from the frozen encoder. The wide version extends only functions beyond 510 content tokens to at most 768 total tokens. The adaptive version can exit after four blocks and forwards uncertain rows through the existing two later blocks. The early and late readouts use the same encoder and the same task labels. All heads are fitted on 12,900 official training-holdout examples, thresholds and routing are selected on 4,292 different examples, and the group-disjoint internal review has 4,294 examples. Exact-text groups use the stored Round 6 partition. Calibration never uses the public split.

| Mode | Public mean F1 | Median full elapsed time | Estimated GFLOPs |
|---|---:|---:|---:|
| Original serial 512+128 | 0.9276 | 23.00 s | 547,405 |
| Exact overlap | 0.9276 | 15.58 s | 547,405 |
| Overlap and learned 512 | 0.9299 | 15.47 s | 544,746 |
| Overlap and learned 768 | 0.9327 | 16.69 s | 564,489 |
| Overlap and adaptive depth | 0.9190 | 13.76 s | 424,841 |

For the 768 candidate the paired 95% interval is +0.25 to +0.85 percentage points in mean F1 against the original. F1 increases from 0.9285 to 0.9307 for vulnerability and from 0.9266 to 0.9347 for MAT. The fixed four-block readout gives mean F1 0.8871. The adaptive version exits early on 22,807 rows and spends 22.4% fewer estimated operations than the original. Its mean F1 loss has paired limits of −1.37 to −0.39 percentage points. [Grouped intervals](public/analysis.json) · [Matched overlap runs](combined/analysis.json).

## Updating some tokens while keeping all context

The production candidate reads a single input, using up to 768 tokens for long functions. All tokens pass through blocks 1–4. In the last two blocks only selected tokens get queries, output projections and feed-forward updates. Unselected states stay at their block-4 values. Every real token remains available as an attention key and value, at its original positions, and the pooling rule still reads the whole state sequence. No encoder weights are retrained. A complete-query control reproduces the original stack exactly.

The fit and calibration groups freeze a selection of roughly 50% of token positions, ranked by the frozen model's token-level classifier directions, with boundary tokens and a minimum of eight tokens. A 65% evenly spaced update policy is the quality alternative. The matched context removal check uses the same 65% updated queries, the same readout and thresholds, and the same final pooling rule.

| Policy at 768 tokens | Mean F1 | Vulnerability F1 | MAT F1 | Median wall | Estimated GFLOPs | Local formula |
|---|---:|---:|---:|---:|---:|---:|
| Full token updates | 0.9327 | 0.9307 | 0.9347 | 16.66 s | 564,489 | 0.9189 |
| Selective updates, about 50% | 0.9311 | 0.9274 | 0.9347 | 15.33 s | 485,224 | 0.9234 |
| Selective updates, about 65% | 0.9329 | 0.9311 | 0.9347 | 15.70 s | 509,236 | 0.9229 |

The 50% policy has the highest measured local score. It reduces estimated operations by 14.0% versus full updates and 11.4% versus the original two-window system. Its mean F1 improvement against the original has a paired 95% interval of +0.06 to +0.69 points. The original vulnerability F1 was 0.9285, so that label loses about 0.11 points at the 50% operating point. The 65% variant improves both labels against the original, and its mean-F1 difference versus the full 768-token model has a paired interval of −0.07 to +0.11 points with 9.8% fewer operations.

Removing the other keys and values while keeping the 65% queries fixed reduces mean F1 from 0.9329 to 0.9098, driven by a vulnerability F1 decrease from 0.9311 to 0.8848. The paired mean F1 difference is +2.12 to +2.52 points in favor of retaining context. The separate pruning variant with recalibrated readout gives 0.9092. This distinguishes the benefit of keeping context visible from the number of positions receiving computation. The full-computation readout trained without tag features gives mean F1 0.9300 and MAT F1 0.9284. [Timed evaluation](tokens_final/analysis.json) · [Matched control](context_ablation/analysis.json).

The competition configuration is **``--policy tokens``**. For the higher-F1 configuration, pass **``--variant query_even_65``**. These decisions carry the original checkpoint plus the corresponding supervised readout and thresholds. Token-depth allocation is related to [PoWER-BERT](https://proceedings.mlr.press/v119/goyal20a.html) and [Mixture-of-Depths](https://arxiv.org/abs/2404.02258).

## Reproduce

The release includes the checkpoint, readout coefficients, thresholds, predictions, benchmark scripts, and original data manifest. Use the repository's tested dependencies. Run the timing commands consecutively on a CUDA host.

```bash
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
python3 nlbse/fetch_inputs.py --training
gzip -dc nlbse/data/split_manifest.csv.gz > nlbse/data/split_manifest.csv
python3 nlbse/round8/tokenizer_speed.py
python3 nlbse/round8/frontier.py tune
python3 nlbse/round8/frontier.py public
python3 nlbse/round8/overlap_speed.py
python3 nlbse/round8/combined.py
python3 nlbse/round8/late_tokens.py tune
python3 nlbse/round8/late_tokens.py public --output nlbse/round8/tokens_final \
  --compare full_even_100 query_saliency_50 query_even_65 prune_even_65
python3 nlbse/round8/matched_context.py
```

Run the predictor with a CSV column named `Function`. The output has binary columns `vulnerability,MAT` in input order. Use `--policy tokens` for the selected 50% policy or add `--variant query_even_65` for the higher-F1 65% variant. The original decisions can be reproduced with overlapping preparation via `--policy exact`. CUDA is required for the learned variants.

```bash
python3 nlbse/round8/predict.py --policy tokens --input functions.csv --output predictions.csv
```
