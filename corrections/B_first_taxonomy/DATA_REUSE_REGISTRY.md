# Data-reuse registry (Paper 2 <-> Paper 3 campaigns)

This file records the overlap between the corrected Paper 2 dataset and the
Paper 3 campaigns (C1, C2, C0) so that the same training run is never counted as
an independent observation in two different analyses.

`lambda_var = 1` is the reference weight used by the 185-run Paper 2 main grid.

## Campaigns

| Campaign | File | N | Status |
|---|---|---|---|
| C1 (`campaign1_baseline`) | `Paper3_prep/results_paper3/campaign1/runs_campaign1.csv` | 50 | complete |
| C2 (`campaign2_sweep_lambda`) | `Paper3_prep/results_paper3/campaign2/runs_campaign2.csv` | 49 | complete |
| C0 (`campagne0_refopt`) | `Paper3_prep/results_paper3/campaign0/` | - | queued |

## Overlap with Paper 2

- **C2 `lambda_var = 1`, seeds 1-6 are byte-identical to C1 seeds 1-6.**
  Same hyperparameters, same code path (`campaign_exp1809.analyze_control_local`
  vs `analyze_control` agree to `|delta| <= 5e-5`).
- **C2 `lambda_var = 0`, seeds 0-4 coincide with the Paper 2 `lambda_var = 0`
  runs** of the `re500_e0.25`, `re700_e2`, `re1000_e1` cells (agreement
  `<= 3.5e-4`).
- **C1 `lambda_var = 1`, seeds 0-4 and C2 `lambda_var = 1`, seeds 0-4 are
  approximately equal to the Paper 2 main-grid runs** at the same `(Re, E*)`
  and seed `0-4` (same protocol; small optimizer fingerprints differ).

## Consequences

1. C1 and C2 must **not** be pooled with the Paper 2 grid as independent samples.
   In §3.6 of the correction report they are listed separately and labelled
   "Paper 3 reuse".
2. The corrected Paper 2 manuscript cites the overlaps as reuse rather than as
   independent validation.
3. Recomputing C1/C2 branch labels with the corrected four-class rule gives
   `P(B_first)` = 0.000 (C1) and 0.571 (C2), consistent with Paper 2's
   `P(B_first) = 0` at `lambda_var = 1` and its rise at `lambda_var = 0`.

## Data-integrity notes (reference campaign)

- `results/01_ReE_star_map/seed_0` is a **misplaced duplicate** of
  `re1000_e2/seed_0` (point_id `re1000_e2`). It is excluded from every canonical
  185-run statistic (the recompute script drops rows whose `rel` has no
  directory separator). Keeping it would give 186 runs.
- The `lambda_var` collector recorded `lambda_var = 0` runs under
  `point_id = "re1000_e1"` (a cosmetic naming collision); the runs themselves are
  correct and are the ones reclassified as `B_first`.
