# REPRODUCE.md — step-by-step verification

Target: a reviewer with **no access to the authors' machine** must be able to
regenerate the quantitative results of the manuscript from this repository alone.

## 0. Requirements

- Python **3.11** (tested: 3.11.4)
- Packages (see `requirements.txt`, exact tested versions in `environment.txt`):

```
numpy==2.2.6
numba==0.61.2
torch==2.12.1+cu126     # required by code/build_controls.py (imports the PINN module)
matplotlib==3.10.9
```

```powershell
py -3.11 -m pip install -r requirements.txt
```

All commands below are run **from the repository root**, so that `config.json`
(relative paths) and `code/` resolve correctly.

## 1. Build the four controls (fast, ~2 s)

```powershell
py -3.11 code\build_controls.py
```

Expected log excerpt — this is the central physical claim of the paper, i.e. all
four lid drives carry **exactly the same actuation energy**:

```
[case] uniform_E025       E0=0.250000 α=—         E_meas=0.25000000 peak=0.5000
[case] B1_E025            E0=0.250000 α=—         E_meas=0.25000000 peak=0.7071
[case] B3_seed0_E025      E0=0.239779 α=1.0210908 E_meas=0.25000000 peak=1.1644
[case] B2_seed5_E025      E0=0.245288 α=1.0095585 E_meas=0.25000000 peak=1.2277
```

Writes `data/lid_series/*.npz` and `results/case_controls_energy.csv`.

## 2. Quick validation run (N = 128)

```powershell
py -3.11 code\run_all.py
```

## 3. Production run reported in the manuscript (N = 256, tol = 1e-9, ~1.5 h)

```powershell
py -3.11 code\run_all.py --N 256 --tol 1e-9 --max-iter 5000000
```

Full batch reported in the paper: 5552 s total. Achieved iteration counts and
residuals per control are recorded in `config.json` under `execution`
(`iters_reached`, `residual_reached`, `walltime_s`) and were:

| control | iterations | final residual |
|---|---|---|
| `uniform_E025` | 218 000 / 482 000 | 9.37e-10 |
| `B1_E025` | 104 000 / 240 000 | 8.20e-10 |
| `B3_seed0_E025` | 280 000 / 548 000 | 1.69e-10 |
| `B2_seed5_E025` | 148 000 / 316 000 | 8.85e-10 |

## 4. Sensitivity studies reported in the manuscript

```powershell
py -3.11 code\sweep_b1_a2.py            # crossing of eps_B1(A2) = eps_uniform(0.5)
py -3.11 code\sweep_energy_matched.py   # iso-energy comparison, k = 1 and k = 2, N = 128/256/512
```

## 5. Figures

```powershell
py -3.11 code\make_figures.py
```

Produces the paper-grade figures (`figures/`, PDF + PNG). The manuscript also
uses the `fig01..fig14` series at the repository root.

## 6. Integrity verification

`code/run_all.py` writes `logs/reproduction_checksums.txt` (SHA256) at the end of
a run, proving that `data/` and `results/` come from exactly this pipeline.

## Expected key numbers (manuscript Table: dissipation ratio)

| control | eps / eps_uniform (N = 256) |
|---|---|
| `uniform_E025` | 1.000 |
| `B1_E025` (A2 = sqrt(0.5)) | 1.188 |
| `B3_seed0_E025` | 0.532 |
| `B2_seed5_E025` | 0.003 |

## 6. Rebuilding the manuscript PDF (optional)

The 14 manuscript figures (`fig01_*.pdf` … `fig14_*.pdf`, PDF + PNG) sit at the
**repository root** and are referenced without a directory prefix, so LaTeX must
be run **from the repository root** (not from `manuscript/`):

```powershell
cd <repository root>
pdflatex -interaction=nonstopmode manuscript\manuscript.tex   # 2 passes
pdflatex -interaction=nonstopmode manuscript\manuscript.tex
```

Tested with MiKTeX: 25 pages, no errors, all 10 embedded figures resolved.
The compiled PDF is committed at `manuscript/manuscript.pdf`.

## Notes for reviewers

- The **temporal branches** (B2, B3) require the `stat_cycles = 48` solver patch
  documented in `PROTOCOL.md`; without it the mean-field decomposition is
  meaningless (see the `R_K` discussion in `README.md`).
- Mesh sensitivity: the *ranking* of the four controls is robust across
  N = 128 / 256 / 512, while the *magnitude* of the ratios is grid-dependent
  (documented in `config.json` → `execution.mesh_note`).
