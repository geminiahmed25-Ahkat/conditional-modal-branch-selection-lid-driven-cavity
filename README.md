# Paper 2 — Case study mécanistique à énergie d'actuation fixée (15/09/2026)

**English title** — *Conditional modal branch selection in physics-informed active control of lid-driven cavity flow: identification, reproducibility, and independent numerical verification*.

Repository: https://github.com/geminiahmed25-Ahkat/conditional-modal-branch-selection-lid-driven-cavity

| Document | Contents |
|---|---|
| [`manuscript/manuscript.pdf`](manuscript/manuscript.pdf) | main article (LaTeX source + PDF) |
| [`manuscript/supplementary_material.pdf`](manuscript/supplementary_material.pdf) | supplementary material |
| [`REPRODUCE.md`](REPRODUCE.md) | **step-by-step reproduction** with expected numbers and runtimes |
| [`DATA_AND_CODE_AVAILABILITY.md`](DATA_AND_CODE_AVAILABILITY.md) | what is included / excluded, and why |
| [`PROTOCOL.md`](PROTOCOL.md) | protocols, solver patch (`stat_cycles`), conventions |
| [`RESULTS.md`](RESULTS.md) · [`RESULTATS_COMPLETS.md`](RESULTATS_COMPLETS.md) | all quantitative results |
| [`CITATION.cff`](CITATION.cff) · [`LICENSE`](LICENSE) | citation metadata (MIT) |

This repository is **self-contained and machine-independent**: all paths in
`config.json` are relative to the repository root and the dependencies of the
companion Paper 1 are vendored in `vendor/paper1/`. A reviewer can reproduce the
study without access to the authors' machine (see `REPRODUCE.md`).

## Objectif

Étude ciblée **Re = 500, AR = 1, E\* = 0.25 FIXÉ pour tous les contrôles** pour
isoler le **rôle de la structure spatio-temporelle du contrôle** sur la
dissipation du fluide, la circulation et le bilan énergétique. **Quatre contrôles
de couvercle** à énergie identique ⟨U_lid²⟩ = 0.25 :

| Contrôle | Forme | Classe | E mesuré | α |
|---|---|---|---|---|
| `uniform_E025` | `U = √0.25 = 0.5` | stationnaire | 0.2500 | — |
| `B1_E025` | `A2·sin(2πx)`, A2=√0.5=0.70710678 | stationnaire 2e harmonique | 0.2500 | — |
| `B3_seed0_E025` | série complète seed 0 (f_2=52%, f_t=45%) | spatio-temporel (B3) | 0.2500 | 1.02109 |
| `B2_seed5_E025` | série complète seed 5 (f_t=100%) | temporel pur (B2) | 0.2500 | 1.00956 |

**Expérience A (archive `results/_v0_archive` + `figures/_v0_archive`)** :
`uniform(1.0)`, `sin_2pi(A2=0.68355)`, `pinn_mean` (moyenne 15 seeds) — validation
LBM du contrôle appris, hors case study à E fixé.

## Diagnostic calculé (champs convergés, normalisés par U_ref, ν = 1/Re)

- **Φ(x, y)** = ν·(2S₁₁² + 2S₂₂² + 4S₁₂²), S₁₁=∂u/∂x, S₂₂=∂v/∂y, S₁₂=½(∂u/∂y+∂v/∂x)
- **Décomposition du taux de déformation** : ε_xx, ε_yy, ε_xy et leurs fractions
- **Topologie** : fonction de courant ψ, vorticité ω = ∂v/∂x − ∂u/∂y
- **Cisaillement pariétal** au couvercle : τ_w(x) = ν·(∂u/∂y)(y=1)
- **Bilan énergétique** : P_lid = −∫ U_lid(récupéré)·τ_w dx vs ε = ∫ Φ dΩ
- **R_K = K_fluct/K** : poids des fluctuations temporelles de la réponse
  (essentiel pour les branches temporelles B2/B3 ; problème de « champ moyen »
  résolu par le patch solveur `stat_cycles=48`, cf. PROTOCOL.md)

## Contenu du dépôt (dépôt GitHub, reviewer-facing)

```
paper2_case_study_repo/
├── config.json          # paramètres du cas (source unique de vérité) + provenance
├── REPRODUCE.md         # protocole de reproduction vérifiable
├── DATA_AND_CODE_AVAILABILITY.md
├── CITATION.cff, LICENSE
├── manuscript/          # manuscript.tex|pdf, supplementary_material.tex|pdf
├── code/                # 11 scripts (build_controls, run_case_study, diagnostics,
│                        #   make_figures, sweep_b1_a2, sweep_energy_matched, run_all, ...)
├── vendor/paper1/       # dépendances Paper 1 vendors (layout d'origine préservé)
│   ├── lbm_mrt_validation/lbm_mrt_pinn_validation.py   # solveur LBM-MRT D2Q9 (numba)
│   ├── PINN_Lid_driven_reviewers.py                    # module PINN (construction UltraPINN, U_lid)
│   ├── seed_models/seed_0..14/model.pt                 # 15 checkpoints PINN (Re500, E*=0.25)
│   ├── ghiau.txt, ghiav.txt                            # données de référence Ghia
│   └── lid_series/                                      # séries de référence Paper 1
├── data/
│   ├── lid_series/          # profils U_lid générés (NPZ) + _E025
│   └── inputs/              # intrants source copiés (fourier_Re500.npz référence Paper 1)
├── results/                 # scalars_*.json, case_study_summary.csv, FIELDS_NOT_IN_REPO.csv
│                            #   (champs .npz de 148 Mo exclus : SHA256 dans FIELDS_NOT_IN_REPO.csv)
├── figures/                 # figs paper-grade (pdf/png)
├── logs/                    # convergence, production_n256.log, checksums
└── fig01..fig14.pdf/png     # figures manuscript (labels EN)
```

## Reproduction

```powershell
py -3.11 -m pip install -r requirements.txt
py -3.11 code\run_all.py                    # N=128, tol 1e-8 (validation rapide)
py -3.11 code\run_all.py --N 256 --tol 1e-9 --max-iter 5000000   # production (≈1.5 h)
```

`run_all.py` écrit en fin de course `logs/reproduction_checksums.txt` (SHA256)
prouvant que data/ et results/ proviennent exactement de ce pipeline.

### Dépendances

- Python **3.11** (env PINN/LBM commun) : numpy, numba, torch (2.12.1+cu126),
  matplotlib, et le solveur LBM-MRT importé depuis
  `PoF_R_lid_driven_paper/lbm_mrt_validation/lbm_mrt_pinn_validation.py`.
- Le script `build_controls.py` importe
  `PoF_R_lid_driven_paper/PINN_Lid_driven_reviewers.py` (garde `__main__`).
- Patch solveur `stat_cycles=None` (optionnel, inoffensif) dans `lbm_mrt_pinn_validation.py`.

## Sources numériques (toutes vendorisées dans ce dépôt)

| Donnée | Chemin dans le dépôt |
|---|---|
| Solveur LBM-MRT (D2Q9, numba) | `vendor/paper1/lbm_mrt_validation/lbm_mrt_pinn_validation.py` |
| Module PINN (construction UltraPINN, `U_lid`) | `vendor/paper1/PINN_Lid_driven_reviewers.py` |
| Modèles PINN seeds (Re = 500, E\* = 0.25) | `vendor/paper1/seed_models/seed_0..14/model.pt` |
| Données de référence Ghia | `vendor/paper1/ghiau.txt`, `vendor/paper1/ghiav.txt` |
| Série fidèle (référence Paper 1) | `vendor/paper1/lid_series/` et `data/inputs/fourier_Re500.npz` |
| Table de campagne (A2, f_2, f_t par seed) | `data/inputs/branch_runs.csv` |
| Table A2 du PINN | `A2_PINN_TABLE` dans `vendor/paper1/lbm_mrt_validation/lbm_mrt_pinn_validation.py` |

## Résultats produits (production N=256, tol 1e-9)

| Contrôle | ε | ε/ε_uniform | K | K_fluct | R_K | Z | P_lid/ε |
|---|---|---|---|---|---|---|---|
| uniform_E025 | 9.498e-3 | 1.000 | 9.35e-3 | 0 | 0 | 4.92 | 0.994 |
| B1_E025 (A2=√0.5) | 1.129e-2 | **1.188** | 1.08e-2 | 0 | 0 | 5.69 | 0.980 |
| B3_seed0_E025 | 5.051e-3 | **0.532** | 3.86e-3 | 1.18e-4 | 0.031 | 2.56 | 0.935 |
| B2_seed5_E025 | 3.098e-5 | **0.003** | 1.55e-5 | 2.31e-4 | **14.96** | 0.016 | 0.365 |

Lecture : à budget E\* identique, la stationnaire 2e harmonique **dégrade** le
bilan (+19 %), la branche mixte le **réduit de ~47 %** (réponse quasi-stationnaire),
la pure temporelle **effondre la dissipation du champ moyen** (−99.7 %) au prix
d'un fort régime fluctuant (R_K≈15 : domaine de validité restreint de la
décomposition champ-moyen). Maillage : N=128 vs N=256 → classement robuste.

Figures manuscrit (labels EN) : `fig_lid_controls`, `fig_composite_4x3`
(4 colonnes × 3 lignes, géométrie exacte AR=1), `fig_quantitative` (a–d),
`fig_tau_w`, + `results/fig_summary_numbers.csv`.