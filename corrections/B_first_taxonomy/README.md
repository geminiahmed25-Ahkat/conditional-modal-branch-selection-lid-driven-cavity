# Correction — four-class branch taxonomy (`B_first`)

This folder contains the corrected version of the Paper 2 analysis. It **supersedes**
the following files at the repository root (kept unchanged for traceability):

| Superseded file (root) | Corrected file (here) |
|---|---|
| `fig03_lambda_branch_probabilities.{png,pdf}` | `figures/fig03_lambda_branch_probabilities.{png,pdf}` |
| `fig04_lambda_entropy.{png,pdf}` | `figures/fig04_lambda_entropy.{png,pdf}` |
| `fig07_loss_ablation.{png,pdf}` | `figures/fig07_loss_ablation.{png,pdf}` |
| `manuscript/manuscript.{tex,pdf}` | `manuscript/manuscript.{tex,pdf}` |
| `manuscript/supplementary_material.{tex,pdf}` | `manuscript/supplementary_material.{tex,pdf}` |

The frozen per-run data are **not** modified; only the branch labels, the two
affected tables and the affected figures/manuscript are corrected.

## What was wrong

1. **Missing branch class (`B_first`).** The three-class scheme
   (`B_1` = second spatial mode `sin(2πx)`, `B_2` = time-dependent, `B_3` = mixed)
   folded into `B_3` every run that was neither `B_1` nor `B_2`. In fact the runs
   at **zero variance regularization (`lambda_var = 0`)** and the **`No L_var`
   ablation** are dominated by the **stationary first harmonic** `sin(πx)`
   (`f_1 = Efrac_0_0 > 0.9`). They form a distinct class, `B_first`.
2. **Threshold wording.** The text quoted `f > 0.70`; the rule actually
   implemented is the dominance rule `f_2 > 2 f_t and f_2 > 0.5`
   (resp. `f_t > 2 f_2 and f_t > 0.5`). Same labels on the 185-run grid, but the
   text must match the code.
3. **Entropy normalization.** The normalized Shannon entropy was divided by
   `ln 3`; with four classes it must be divided by `ln 4`.

Corrected taxonomy:

```
B_first       f_1 > 0.5                      f_1 = Efrac_0_0   (sin(pi x))
B1_Mode2      f_2 > 2*f_t and f_2 > 0.5      f_2 = Efrac_1_0   (sin(2 pi x))
B2_temporal   f_t > 2*f_2 and f_t > 0.5
B3_mixed      otherwise
```

## Impact

The 185-run reference campaign (`lambda_var = 1`) contains **no** `B_first`
(`f_1 <= 0.13`): its published statistics (`B_1` 42.2 %, `B_2` 43.8 %,
`B_3` 14.1 %; 0/185 discrepancies) are unchanged, and all main conclusions hold.
The correction changes the interpretation of the `lambda_var` sweep and the loss
ablation: at `lambda_var = 0` the model deterministically selects `B_first`
(`sin(πx)`), not a "mixed" control; any positive `lambda_var` suppresses
`B_first`. See `CORRECTION_REPORT.md` for every corrected number.

## Contents

- `results/corrected_*.csv` — all corrected tables (main grid, `lambda_var`,
  conditional-accessibility, ablation, seeds, energy sweep, global summary).
- `figures/fig03|fig04|fig07.*` — corrected figures.
- `manuscript/` — corrected manuscript and supplementary (LaTeX + compiled PDF).
- `branch_labels.py` — corrected four-class classifier (`behavior_name4`).
- `01_recompute_taxonomy.py`, `02_make_figures.py` — reproducible recomputation.
- `CORRECTION_REPORT.md` — full report (what was wrong, all corrected numbers).
- `DATA_REUSE_REGISTRY.md` — training-run reuse disclosure (Paper 2 <-> Paper 3).
- `NOTICE_TO_COAUTHORS.md`, `LETTER_TO_EDITOR.md` — disclosure drafts.

## Reproduce

```powershell
# Python (numpy / pandas / matplotlib):
py -3.11 01_recompute_taxonomy.py   # regenerates results/corrected_*.csv
py -3.11 02_make_figures.py         # regenerates figures/fig03, fig04, fig07
# LaTeX (the manuscript compiles with pdflatex):
cd manuscript; pdflatex manuscript.tex; pdflatex manuscript.tex
```
