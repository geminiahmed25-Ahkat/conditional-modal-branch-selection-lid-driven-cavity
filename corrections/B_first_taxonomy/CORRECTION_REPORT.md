# Paper 2 - Correction report (clean corrected package)

**Scope.** This package is a self-contained, corrected version of the Paper 2
study *"Conditional modal branch selection in physics-informed active control of
lid-driven cavity flow: identification, reproducibility, and independent numerical
verification"*. It contains the corrected code, the recomputed result tables, the
corrected figures, and the corrected LaTeX manuscript and supplementary material.

**Status of the original.** The original Paper 2 directory
(`paper_2_case_study_15_09_2026/`) was **not modified** except for the
figure-reproduction bundle script, which was extended to match the corrected
class definition. All corrections are reproduced from scratch by the scripts in
`01_code/` from the frozen per-run data.

---

## 1. What was wrong

1. **Missing branch class (`B_first`).** The classifier grouped every realization
   that was neither second-harmonic dominated (`B_1`) nor time-dependent dominated
   (`B_2`) into a single "mixed" class `B_3`. In reality, the runs obtained with
   **zero variance regularization** (`lambda_var = 0`) are not mixed at all: they
   are dominated by the **stationary first harmonic** (`f_1 = Efrac_0_0 > 0.9`).
   The three-class taxonomy therefore labelled a physically distinct attractor as
   "mixed".

2. **Wrong threshold values in the manuscript text.** The main text stated
   `f_2 > 0.70` and `f_t > 0.70`. The rule actually implemented in
   `branch_labels.py` (and used for every label) is the dominance rule
   `f_2 > 2 f_t and f_2 > 0.5` (resp. `f_t > 2 f_2 and f_t > 0.5`). The two rules
   happen to give identical labels on the 185-run reference grid, but the
   manuscript text did not describe the rule that was used.

3. **Entropy normalization.** The normalized Shannon entropy was normalized by
   `ln 3` (three classes). With four classes it must be normalized by `ln 4`.

**Impact on the reference campaign.** The 185-run reference campaign was run at
`lambda_var = 1` and contains **no** first-harmonic realization
(`f_1 <= 0.12` everywhere). Its published statistics
(`B_1` 42.2 %, `B_2` 43.8 %, `B_3` 14.1 %; 0/185 label discrepancies) are therefore
**unchanged**. The error only affects the `lambda_var` sweep and the loss ablation,
where `lambda_var = 0` and the `No L_var` ablation were mislabeled as `B_3`.

---

## 2. Corrected taxonomy

```
B_first       f_1 > 0.5
B1_Mode2      f_2 > 2*f_t and f_2 > 0.5
B2_temporal   f_t > 2*f_2 and f_t > 0.5
B3_mixed      otherwise
```
with `f_1 = Efrac_0_0`, `f_2 = Efrac_2_0`, `f_t = sum of time-dependent fractions`,
and entropy `H_N = -1/ln(4) * sum_k p_k ln p_k`.
Implemented in `01_code/branch_labels.py::behavior_name4`.

---

## 3. Corrected numbers

Recomputed by `01_code/01_recompute_taxonomy.py`; outputs in `02_results/`.

### 3.1 Reference campaign (185 runs, `lambda_var = 1`) - unchanged
| Branch | N | Frequency | mean f_2 | mean f_t |
|---|---|---|---|---|
| B1 | 78 | 42.2 % | 85.3 % | 7.4 % |
| B2 | 81 | 43.8 % | 7.5 % | 88.1 % |
| B3 | 26 | 14.1 % | 34.8 % | 43.9 % |
| B_first | 0 | 0.0 % | - | - |

`H_N = 0.722` (ln4 normalization; the old ln3 value was 0.912).

### 3.2 `lambda_var` sweep (per cell, 5 seeds) - corrected
| Re | E* | lambda | P(B1) | P(B2) | P(B_first) | P(B3) | H_N | dominant |
|---|---|---|---|---|---|---|---|---|
| 500 | 0.25 | 0 | 0.0 | 0.0 | **1.0** | 0.0 | 0.000 | B_first |
| 500 | 0.25 | 0.25 | 0.6 | 0.4 | 0.0 | 0.0 | 0.485 | B1 |
| 500 | 0.25 | 0.5 | 0.4 | 0.2 | 0.0 | 0.4 | 0.761 | B1/B3 |
| 500 | 0.25 | 1 | 0.4 | 0.2 | 0.0 | 0.4 | 0.761 | B1/B3 |
| 500 | 0.25 | 2 | 0.4 | 0.4 | 0.0 | 0.2 | 0.761 | B1/B2 |
| 500 | 0.25 | 5 | 0.4 | 0.6 | 0.0 | 0.0 | 0.485 | B2 |
| 700 | 2 | 0 | 0.0 | 0.0 | **1.0** | 0.0 | 0.000 | B_first |
| 700 | 2 | 0.25 | 0.0 | 0.0 | **1.0** | 0.0 | 0.000 | B_first |
| 700 | 2 | 0.5 | 0.0 | 0.6 | 0.0 | 0.4 | 0.485 | B2 |
| 700 | 2 | 1 | 0.2 | 0.4 | 0.0 | 0.4 | 0.761 | B2/B3 |
| 700 | 2 | 2 | 0.6 | 0.2 | 0.0 | 0.2 | 0.685 | B1 |
| 700 | 2 | 5 | 0.6 | 0.2 | 0.0 | 0.2 | 0.685 | B1 |
| 1000 | 1 | 0 | 0.0 | 0.0 | **1.0** | 0.0 | 0.000 | B_first |
| 1000 | 1 | 0.25 | 0.0 | 0.6 | 0.0 | 0.4 | 0.485 | B2 |
| 1000 | 1 | 0.5 | 0.4 | 0.4 | 0.0 | 0.2 | 0.761 | B1/B2 |
| 1000 | 1 | 1 | 0.4 | 0.4 | 0.0 | 0.2 | 0.761 | B1/B2 |
| 1000 | 1 | 2 | 0.4 | 0.4 | 0.0 | 0.2 | 0.761 | B1/B2 |
| 1000 | 1 | 5 | 0.4 | 0.6 | 0.0 | 0.0 | 0.485 | B2 |

Pooled over the three operating points (15 seeds per lambda):
| lambda | B1 | B2 | B_first | B3 | H_N |
|---|---|---|---|---|---|
| 0 | 0.000 | 0.000 | **1.000** | 0.000 | 0.000 |
| 0.25 | 0.200 | 0.333 | 0.333 | 0.133 | 0.954 |
| 0.5 | 0.267 | 0.400 | 0.000 | 0.333 | 0.783 |
| 1 | 0.333 | 0.333 | 0.000 | 0.333 | 0.792 |
| 2 | 0.467 | 0.333 | 0.000 | 0.200 | 0.753 |
| 5 | 0.467 | 0.467 | 0.000 | 0.067 | 0.643 |

**Interpretation changed.** At `lambda_var = 0` the optimizer does **not** collapse
to "unstructured mixed controls"; it deterministically selects the lowest-order
stationary mode (`sin(pi x)`, i.e. `B_first`). Any positive `lambda_var` suppresses
`B_first`. This strengthens the paper's message: the variance penalty actively
*excludes* the physically low-dissipation fundamental mode.

### 3.3 Conditional accessibility campaign (54 runs) - corrected
| exp | N | B1 | B2 | B_first | B3 | dominant |
|---|---|---|---|---|---|---|
| A (`m=1` init) | 12 | 0 | 0 | **12** | 0 | B_first |
| B (`m=2` init) | 12 | 12 | 0 | 0 | 0 | B1 |
| C (`init_k1`,`init_k2`) | 24 | 19 | 5 | 0 | 0 | B1 |
| D (default) | 3 | 1 | 1 | 0 | 1 | B1/B2/B3 |
| T (`tau=2`) | 3 | 0 | 3 | 0 | 0 | B2 |

### 3.4 Loss ablation (`Re=500`, `E*=0.25`) - corrected
| Configuration | E_r | f_1 | f_2 | f_t | Class |
|---|---|---|---|---|---|
| Baseline | 0.2534 | 0.0 | 92.2 | 1.3 | B1 |
| No L_var | 0.2582 | **90.4** | 0.2 | 5.1 | **B_first** |
| No L_Re | 0.2534 | 0.0 | 92.2 | 1.3 | B1 |
| No L_diss | 0.2533 | 0.0 | 92.3 | 1.2 | B1 |
| No L_var,L_Re | 0.2582 | **90.4** | 0.2 | 5.1 | **B_first** |
| No L_var,L_diss | 0.2578 | **90.5** | 0.3 | 5.0 | **B_first** |
| lambda/2 | 0.2546 | 0.2 | 89.3 | 1.5 | B1 |
| 2*lambda | 0.2598 | 0.1 | 92.8 | 1.0 | B1 |

The `No L_var` "modal-class shift" is now identified as `B_1 -> B_first`, not
`B_1 -> mixed`.

### 3.5 Seed study (10 seeds, `Re=500`, `E*=0.25`) - unchanged
4 `B_1`, 4 `B_2`, 2 `B_3`; **0 `B_first`** (`f_1 < 0.02` for all seeds).

### 3.6 Global summary (all canonical runs)
| source | N | B1 | B2 | B_first | B3 |
|---|---|---|---|---|---|
| P2 main grid | 185 | 0.422 | 0.438 | 0.000 | 0.141 |
| P2 lambda_var | 90 | 0.289 | 0.311 | 0.222 | 0.178 |
| P2 exp09 | 54 | 0.593 | 0.167 | 0.222 | 0.019 |
| C1 (Paper 3) | 50 | 0.620 | 0.320 | 0.000 | 0.060 |
| C2 (Paper 3) | 49 | 0.245 | 0.143 | 0.571 | 0.041 |

(C1/C2 are shown for the reuse registry only and are **not** Paper 2 results.)

---

## 4. Manuscript changes

File `04_manuscript/manuscript.tex` (corrected) and
`04_manuscript/supplementary_material.tex` (corrected):

- Abstract: "three" -> "four" empirical classes; `B_first` introduced; added the
  statement that removing the variance weight selects `B_first`.
- Abbreviation table: added `B_first` and `f_1`.
- Sec. "Branch classification": four-class rule with explicit thresholds
  (dominance 2x + 0.5), `H_N` normalized by `ln 4`, and a note that the previous
  three-class scheme folded `B_first` into `B_3`.
- Global landscape: note that the reference campaign has no `B_first` (`f_1<=0.13`).
- Observable space: note that `B_first` is empty at `lambda_var = 1`.
- `lambda_var` section: table extended to four classes; `H_N` recomputed; text
  rewritten (λ=0 -> deterministic `B_first`, single-lobe `sin(pi x)`).
- Energy sweep: note `f_1 <= 8 %` (no `B_first`).
- Seed study: note `f_1 < 0.02` (no `B_first`).
- Loss ablation: table extended with `f_1` and Class; text rewritten
  (`B_1 -> B_first`).
- Sec. "Optimization bias vs fluid mechanics": `No L_var` shift now `B_first`.
- Discussion: `B_3` no longer attributed to `lambda_var = 0`; new paragraph on the
  `B_first` / `B_1` pair and its control by a single loss weight.
- LBM focused comparison: linked the pure first-harmonic profile (ii) to `B_first`
  and noted that the non-selected branch is the energetically preferable one.
- Conclusions: items 1 and 2 rewritten for the four-class taxonomy.
- Multi-seed audit revalidation: threshold rule updated to the four-class rule
  (still 0/185 discrepancies).
- Supplementary S3: table + rule updated to four classes; S4 note that all
  `lambda_var = 0` runs are `B_first`; S5 energy-sweep note and seed-study note.

The manuscript compiles with `pdflatex` (26 pages) with no undefined references.

## 5. Figures

Corrected in `03_figures/` and `04_manuscript/`:

- `fig03_lambda_branch_probabilities` - now a 4-class heatmap (B1, B2, B_first, B3)
  with annotated cell values.
- `fig04_lambda_entropy` - recomputed with `ln 4` normalization.
- `fig07_loss_ablation` - now three panels (`f_1`, `f_2`, `f_t`).

Unchanged (no `B_first` realization in the corresponding campaigns):
`fig01`, `fig02`, `fig05`, `fig06`, `fig08`-`fig14`.

## 6. Reuse note for co-authors and the editor

The Paper 3 campaigns C1/C2 re-use training runs that are also part of Paper 2's
design space; see `DATA_REUSE_REGISTRY.md`. They must not be counted as
independent observations with respect to Paper 2, and the corresponding overlap is
disclosed in the corrected manuscript's reproducibility discussion.

## 7. Deferred to Paper 3

**Trajectory analysis of the first-harmonic exit (step B) is deferred to Paper 3.**
The conditional-accessibility experiment `expC_init` (init `k1`/`k2`, 24 runs,
`results/09_conditional_accessibility/expC_init/`) stores only the final state
(`features.json`) and per-epoch loss/energy scalars (`training_history.csv`); it
contains **no per-epoch modal decomposition**, so the leaving-the-first-harmonic
trajectory cannot be reconstructed without an instrumented re-run.

The only stored per-epoch modal traces are `trajectory_modal.csv`
(columns `f1`, `mode2_fraction`, `temporal_fraction`, sampled every 50 epochs,
61 points) for **`expD_traj`** (init `default`, `re500_e0.25`, seeds 4/7/8). These
show the control starting near the first harmonic (`f1` 0.33 at epoch 50 for
seed 4) and leaving it (`f1` -> 0.003, `temporal_fraction` -> 0.64 by epoch 3000).
Paper 3 should instrument `expC_init` the same way to analyze the exit
quantitatively.

This item does not affect the corrected Paper 2 results; it is recorded here only
so the open question is not lost.
