# Which omitted tokens explain the MAT gain?

This follow-up uses the original CodeT5+ checkpoints, frozen first-pass scores, thresholds, and second-window row IDs. Every masked comparison retains the original window shapes, token positions, and boundary tokens. The script verifies that it reproduces the prior comment-only decisions. It analyzes all 30,655 public functions and all 31,104 functions of the supervised Chromium, QEMU, and FFmpeg holdout, with 238 and 156 routed windows respectively.

| Second-window diagnostic | Public MAT F1 | Project holdout MAT F1 |
|---|---:|---:|
| First pass alone | 0.9048 | 0.9301 |
| Full second window | 0.9266 | 0.9390 |
| Only comments already visible to the first pass | 0.9048 | 0.9301 |
| Only omitted comment tokens | 0.9257 | 0.9390 |
| Only omitted comment tokens after masking the four tags | 0.9038 | 0.9302 |
| Physically packed omitted comment tokens | 0.9266 | 0.9383 |
| Four-tag rule in the identical omitted span | 0.9266 | 0.9395 |
| Four-tag rule across all comments | 0.9336 | 0.9455 |

The four case-insensitive whole-word tags are TODO, FIXME, XXX, and HACK, following the MAT approach described by [Russo et al.](https://arxiv.org/html/2501.15934v1#S3). The rule for the same span retains the original first-pass decision when it is positive. The full-comment rule predicts the MAT label directly. Both diagnostics retain the original vulnerability predictions.

The omitted span contains at least one of the tags in all 19 rescued public positives and all 14 rescued project positives. Relative to the full second window, the same-span rule changes zero public decisions and clears one project false positive. The public compact input matches every decision of the full second window. The compact input misses one true positive on the project holdout. Estimated dense operations for the second pass drop from 2,659 to 1,052 GFLOPs and from 1,743 to 517 GFLOPs respectively. These estimates cover the additional pass and use the padded batch shapes. Run time of the complete pipeline needs its own measurement.

The paired 95% intervals for MAT F1 of omitted-only minus tag-masked input are +1.22 to +3.23 percentage points on public validation and +0.44 to +1.36 on the project holdout, resampled by project and file. The experiment supports recovery of annotation tags lost at truncation. Untagged human-identified technical debt requires separate evaluation.

## Reproduce

Start with the archived round 6 results and the two original checkpoints restored. The project checkpoint can be regenerated following the [project experiment recipe](../README.md). Use the environment recorded in the release and run from its root.

```bash
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
python3 nlbse/round6/omitted_comments.py public --output nlbse/round6/omitted_replay/public
python3 nlbse/round6/omitted_comments.py project --output nlbse/round6/omitted_replay/project
```

The [public report](public/analysis.json) and [project report](project/analysis.json) record data alignment, hashes, decisions, and paired comparisons. Per-variant metrics and ordered predictions are in adjacent directories.
