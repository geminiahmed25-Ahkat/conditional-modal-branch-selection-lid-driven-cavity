# Data and Code Availability

This repository is the reference implementation and data release for:

> **Paper 2 — Focused fixed-energy comparison of the physical consequences of
> competing control classes** (lid-driven cavity, Re = 500, AR = 1,
> actuation-energy budget E* = 0.25 fixed for all four controls).

## What is included here

| Item | Location | Status |
|---|---|---|
| Manuscript (LaTeX + PDF) | `manuscript/` | included |
| Supplementary Material (LaTeX + PDF) | `manuscript/supplementary_material.*` | included |
| Analysis / solver driver code (11 scripts) | `code/` | included |
| Vendored dependencies (LBM-MRT solver, PINN module, 15 seed models, Ghia reference data) | `vendor/paper1/` | included |
| Lid-drive series (NPZ) and reference Fourier series | `data/` | included |
| Quantitative results (JSON / CSV) and convergence logs | `results/`, `logs/` | included |
| Publication figures (PDF + PNG) | `figures/`, `fig01..fig14` | included |
| Converged raw LBM fields (NPZ, 148 MB) | `results/fields_*.npz` | **excluded from git** (see below) |

## Why the raw fields are not in git

The converged velocity/strain-rate fields are large binary blobs
(52 NPZ files, 148 MB) whose *scientific content is fully captured* by the
scalar JSON/CSV files already in `results/`. To keep the repository light and
reviewable, they are excluded from version control and their integrity is
recorded by SHA256 in:

```
results/FIELDS_NOT_IN_REPO.csv
```

They are available on request, and can be regenerated bit-for-bit by running
the reproduction pipeline (`REPRODUCE.md`), which writes
`logs/reproduction_checksums.txt` at the end of the run.

## Reproducibility

The repository is **self-contained**: no absolute, machine-specific paths remain
(`config.json` uses paths relative to the repository root, resolved by the code
via `ROOT = dirname(dirname(__file__))`). The dependencies of the companion
Paper 1 (LBM-MRT solver `lbm_mrt_pinn_validation.py`, PINN module
`PINN_Lid_driven_reviewers.py`, the 15 PINN seed checkpoints and the Ghia
reference data) are vendored in `vendor/paper1/`, preserving the directory
layout expected by the code.

Full step-by-step instructions, including expected runtimes and verification
checksums, are given in `REPRODUCE.md`.

## Provenance of every number in the manuscript

`config.json` is the single source of truth: it records the case definition
(Re, E*, AR), the provenance of each amplitude and seed, the solver settings
used for the production runs (N = 256, tolerance 1e-9, max 5e6 iterations,
`stat_cycles = 48`), the achieved iteration counts and residuals, wall-clock
times, the mesh-refinement check (N = 128 vs N = 256 vs N = 512), and the
energy-matched sweeps (`sweep_b1_a2.py`, `sweep_energy_matched.py`).
`RESULTS.md` and `RESULTATS_COMPLETS.md` tabulate the values reported in the
manuscript, and `PROTOCOL.md` documents the mean-field statistics patch
(`stat_cycles`) required for the temporal branches.
