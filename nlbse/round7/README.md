# Tag-free SATD and the cost of a compact second pass

## Full-pipeline timing

Six interleaved timing pairs used the same FP16 CodeT5+, thresholds, RTX 4090, input order, batch scheduler, and 30,655 public rows. Both modes were warmed up first. The original mode reads a contiguous 128-token window and the compact mode packs only the omitted comment tokens within that span. Both routed 238 functions and reproduced all 61,310 released decisions. Second-pass wall time includes batch assembly, input transfer, model execution, and transfer of scores. The end-to-end time includes tokenization, scanning, both passes, and selection.

| Measurement | Contiguous | Compact |
|---|---:|---:|
| Mean second input, including boundary tokens | 128.00 | 17.79 |
| Extra-pass estimated GFLOPs | 2,659 | 1,052 |
| Total estimated GFLOPs | 547,405 | 545,798 |
| Median second-pass wall time | 91.92 ms | 88.79 ms |
| Median total wall time | 22.941 s | 23.011 s |
| Public mean F1 | 0.927557 | 0.927557 |

The compact variant reduced estimated second-pass work by 60.4% and total work by 0.29%. The second pass was faster in all six pairs, with a median paired improvement of 3.29 ms. The whole pipeline was faster in two of six pairs. The paired wall differences (original minus compact) were 23, −190, −54, 541, −163, and −361 ms. The timing evidence supports only a small second-pass wall-time effect on this host. [All timing trials and scores](timing/analysis.json).

## Independently annotated comments without MAT tags

[CppSATD](https://zenodo.org/records/15284981) supplies human annotations and natural adjacent C++ context. The five CppSATD repositories are absent from the official NLBSE supervised training set by project name. [Tesoro](https://github.com/NamCyan/tesoro) supplies human annotations and full Java methods. The strict input cohort required one alignable target and excluded inputs with any of the four MAT tags. CppSATD retained 16,955 unique comments after filtering. On strict Tesoro the 512-token cutoff omitted part of just one example. To obtain a useful separate Tesoro comparison, we removed all other comments, preserved line breaks, and kept the uniquely aligned target at its corresponding position in the remaining code. There were 2,138 deduplicated, tag-free Java inputs after also removing normalized comment overlap with CppSATD. Raw strict Java results are preserved.

Labels identify the target comment rather than vulnerabilities or the entire file. The 512-token cap is the primary cutoff. The 256 and 128 caps are stress tests, with a different eligible cohort at each cutoff. Each comparison uses the same examples at its cutoff. Only inputs with at least one target token beyond the cutoff are scored. We compare a single head, an already-seen fragment of the same width, the targeted contiguous second window, and compact omitted comment tokens.

| 512-token study | CppSATD, 218 rows / 20 positive | Tesoro, 870 rows / 21 positive |
|---|---:|---:|
| Frozen MAT at its published threshold, head → full-window F1 | 0.000 → 0.000 | 0.000 → 0.000 |
| Frozen MAT, head → full-window average precision | 0.088 → 0.083 | 0.024 → 0.096 |
| Human-label probe, seen control → compact F1 | 0.227 → 0.233 | 0.046 → 0.052 |
| Human-label probe, seen control → compact average precision | 0.182 → 0.450 | 0.024 → 0.024 |
| Human-label probe, max → ignore head when fully omitted F1 | 0.233 → 0.273 | 0.052 → 0.167 |

The C++ ranking improved most with a probe trained on human-annotated C++ comments. It used frozen CodeT5+ embeddings and one logistic classifier, with the held-out project excluded from fitting. A second experiment trained on all retained CppSATD examples and transferred to Tesoro without fitting on any Java labels. The head and the second-pass scores use the same probe. The lexical comparison trains char n-gram TF-IDF and logistic regression on the same project splits. At the 512 cap the lexical model achieved F1 0.453 on CppSATD and the embedding probe scored 0.371 when both use only the compact comment. At the 256 cap on CppSATD, max-combining the compact comment gave F1 0.291 versus 0.221 when using already-seen inputs (paired 95% interval +3.20 to +11.28 percentage points).

The frozen MAT output is evaluated at its stored threshold of 0.50. A scalar recalibration control maximizes F1 only on different CppSATD projects before testing. At 512, CppSATD recalibrated F1 was 0.144 head-only, 0.176 against the seen control, and 0.181 with the full extra window. The difference from the seen control is small. Tesoro's recalibrated result depends on width and the input variant. Average precision measures ranking irrespective of that operating threshold.

## Testing the score combination

A follow-up replaced the max combination by the compact-comment score only when zero tokens of the target were visible. It kept the max on partially visible targets. The rule uses token positions and no fitted parameters. At 512 resolution it removed 19 false positives on CppSATD without losing a true positive and removed 527 false positives on Tesoro at the cost of four true positives. Its paired F1 differences from max were +1.84 to +6.85 points (CppSATD) and +5.35 to +24.48 (Tesoro). These are exploratory, same-input comparisons. CppSATD and Tesoro have different annotation and sampling procedures. CppSATD uses prior lexical patterns to nominate candidates, and Tesoro evaluation here uses a single-comment view. The estimates characterize these routed cohorts. They leave pretraining overlap and general-population SATD rates unresolved.

Probabilities, row IDs, labels and source hashes, first and last target token indices, and all variants are in the result folders. Bootstraps group by project and file on C++ and source function ID on Java. [External metrics](external_aligned/analysis.json) · [Threshold controls](external_aligned/calibration.json) · [Score combination and paired intervals](external_aligned/ranking.json).

## Reproduce

Run from the release root. The frozen NLBSE checkpoint and data are checked against the published digests. The independent datasets are fetched from the pinned [manifest](data/manifest.json). The experiment tables record row identifiers, labels, and source hashes. The rerun downloads the original datasets from their authors.

```bash
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
python3 nlbse/fetch_inputs.py --training
python3 nlbse/round7/benchmark_compact.py --repeat 6 --output nlbse/round7/replay_timing
python3 nlbse/round7/external_satd.py --output nlbse/round7/replay_external
python3 nlbse/round7/external_satd.py --isolate-tesoro --output nlbse/round7/replay_aligned
python3 nlbse/round7/recalibrate.py --input nlbse/round7/replay_aligned
python3 nlbse/round7/ranking.py --input nlbse/round7/replay_aligned
```
