# Round 3 · four local NLBSE ideas

Compare on the same 18,444 official-train holdout functions used for the previous selection. Calibration uses 3,042 separate exact-text-grouped rows. New small heads train on the existing stratified 20,000-row fit subset with frozen encoders. The experiments use the same checked data and checkpoints as round 2. Both heads of every variant predict in vulnerability/MAT order.

| Idea | Implementation | Matched control |
|---|---|---|
| Simple | Six blocks. Separate decision thresholds for up to 128, 129–256 and more than 256 input tokens. Each calibration estimate is pulled half way toward the global threshold. Sparse positive counts use the global threshold. | Six blocks and global thresholds |
| Quality-oriented | Twelve blocks with the original first token window. Overlength functions also contribute a trailing 512-token window using the same encoder and head. Choose class-specific blending coefficients on calibration. | Twelve blocks, first window |
| Split | One six-block network. Pool code and comment token states separately and learn a small residual classifier branch per class. Both branches use the shared encoding and its classification token. | The same six blocks and original head |
| Early exit | One twelve-block network. Distill a two-output intermediate head after block six. Run the remaining six blocks only for rows with a low joint margin. Choose the most aggressive tested gate within 0.005 mean F1 of full depth on calibration. | The same twelve blocks with full depth |

All four consume the function text alone. The comment scanner handles C-style comments and the common literal forms. It is lexical. The role and early-exit branches were trained with weighted BCE and soft labels from their respective frozen checkpoint on fitting rows. No public-test labels are used. All measured GPU variants use the same eager attention, FP16, and length-sorted 32-row batches. They profile on prepared tokens, so the timing excludes tokenization and the role mask scan. Compare operation counts and within-round times. The organizer's final T4 setup is distinct.

Run from the experiment root on a host that has already reproduced the first two rounds.

    bash nlbse/round3/run.sh

The previously prepared submission remains a separate artifact. Per-run JSON and ordered predictions are under runs. Paired intervals are based on 2,000 bootstrap draws of exact-function groups.

The first early-exit grid stopped at 0.85 before the no-exit control. When it selected no exits, we extended the grid with high-confidence thresholds using the same calibration data. If all active gates fail the quality criterion, the follow-up measures the 0.85 gate as a separate diagnostic. This reuses the saved intermediate head and its fixed full-depth encoder.

The context variant is also run once on all 30,655 public rows using its frozen blending weights and thresholds. This additional run checks quality. Runtime is recorded without the operation profiler. The matched comparison tests both threshold settings on both score vectors as a diagnostic, with no refitting.

    .venv/bin/python -u nlbse/cuda_launch.py nlbse/round3/public_context.py

The original twelve-block public prediction file is the matched reference. The public comparison uses paired resampling by project and file.
