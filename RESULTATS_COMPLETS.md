# Résultats complets — Campagne de validation Paper 2 (case study)

Date de génération : 2026-09-17 · Racine : `paper_2_case_study_15_09_2026/`

Ce document regroupe **l'ensemble des résultats de la campagne de validation**
(case study mécanistique Re=500, AR=1) : conditions opératoires, réglages
numériques, tableaux chiffrés par campagne, temps de calcul et éléments de
reproductibilité (scripts, commandes, séries, checksums).

---

## 1. Contexte et question scientifique

À énergie d'actuation fixée **E\* = ⟨U_lid²⟩**, comment l'organisation modale du
contrôle (uniforme / harmonique stationnaire / mixte temporel / temporel pur)
réorganise-t-elle circulation, taux de strain, cisaillement pariétal et
dissipation ?

- Point d'opération : **Re = 500**, **AR = 1**, champs normalisés par `U_ref`.
- Énergie du point de fonctionnement : **E\* = 0.25** (`U = √0.25 = 0.5` pour le
  lid uniforme ; `A2 = √(2·E\*) = √0.5 = 0.70710678` pour les harmoniques).
- Convention : **ν = 1/Re = 0.002** (non-dimensionnel).

---

## 2. Conditions opératoires et réglages numériques

| Paramètre | Valeur | Source |
|---|---|---|
| Solveur | LBM-MRT D2Q9, numba | `run_case_study.py` / `lbm_mrt_pinn_validation.py` |
| Re | 500 | config.json |
| AR | 1.0 | config.json |
| U_ref | 0.05 | config.json `lbm.U_ref` |
| Énergie cible E\* | 0.25 | config.json `case.energy_budget` |
| Grilles | N = 128 / 256 / 512 | config.json |
| Tolérance prod. | 1e-9 (N=256, N=512) | config.json `tol_production` |
| Tolérance validation | 1e-8 (N=128) | config.json `tol_default` |
| max_iter prod. | 5 000 000 | config.json `max_iter_production` |
| Période forcée (B2/B3) | 256 pas / 2.0 t | config.json `period_forced` |
| Statistique contrôles temporels | 48 cycles verrouillés (stat_cycles), stat_iters=12288 | patch solveur |
| Critère de convergence | max|Δu| < tol, 2 contrôles consécutifs, check / 2000 pas | solveur |

Diagnostics (convention `run_case_study.py`) :
- Φ = ν(2S11² + 2S22² + 4S12²) = 2ν S:S, avec S11=∂u/∂x, S22=∂v/∂y, S12=½(∂u/∂y+∂v/∂x).
- ε = ∫Φ dA (champ moyen) ; ε_xx=2ν∫ux², ε_yy=2ν∫vy², ε_xy=ν∫(uy+vx)².
- K = ½⟨|u|²⟩, K_fluct (fluct. temporelles), R_K = K_fluct/K.
- Z = vorticité ∫|ω| dA ; P_lid = −∫ U_lid(récupéré)·τ_w dx ; P_lid/ε = bilan.
- τ_w(x) = ν·(u_{N−1}−u_{N−2})/dx.

---

## 3. Campagne production — 4 classes de contrôle (N=256, tol 1e-9)

Séries PINN : seeds 0..14 de `results/01_ReE_star_map/re500_e0.25/seed_*/model.pt`
(`UltraPINN(basis='fourier')`, x 256 pts, t ∈ [0,2] en 256 pas) ; construction dans
`code/build_controls.py` (normalisation α = √(E/⟨U²⟩_x,t)).

| Contrôle | Type | Construction | E mesuré | α |
|---|---|---|---|---|
| uniform_E025 | stationnaire | U(x) = 0.5 | 0.2500 | 1.0 |
| B1_E025 | stationnaire | A2·sin(2πx), A2=√0.5 | 0.2500 | 1.0 |
| B3_seed0_E025 | temporal | série complète seed 0 | 0.2500 | 1.02109 |
| B2_seed5_E025 | temporal | série complète seed 5 | 0.2500 | 1.00956 |

### 3.1 Table des grandeurs (production N=256)

| Contrôle | ε | ε/ε_uniform | ε_xx | ε_yy | ε_xy | frac_xy | K | K_fluct | R_K | Z | P_lid | P_lid/ε |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Uniform (U=0.5) | 9.498e-3 | 1.000 | 1.166e-3 | 1.155e-3 | 7.178e-3 | 0.756 | 9.352e-3 | 0 | 0 | 4.916 | 9.441e-3 | 0.994 |
| B1 (A2 sin 2πx) | 1.1285e-2 | **1.188** | 1.270e-3 | 1.265e-3 | 8.750e-3 | 0.775 | 1.079e-2 | 0 | 0 | 5.689 | 1.106e-2 | 0.980 |
| B3 (seed 0) | 5.051e-3 | **0.532** | 7.867e-4 | 7.836e-4 | 3.481e-3 | 0.689 | 3.864e-3 | 1.180e-4 | 0.031 | 2.555 | 4.721e-3 | 0.935 |
| B2 (seed 5) | 3.098e-5 | **0.003** | 5.844e-6 | 5.828e-6 | 1.931e-5 | 0.623 | 1.545e-5 | 2.311e-4 | **14.96** | 0.0158 | 1.130e-5 | 0.365 |

Source : `results/case_study_summary.csv`. ε du champ MOYEN ; pour B2 la
dissipation instationnaire réelle est portée par les fluctuations (R_K ≫ 1).

### 3.2 Convergence et temps CPU (N=256 vs N=128)

| Contrôle | iters N=128 | iters N=256 | résid. final N=256 | wall N=128 (s) | wall N=256 (s) |
|---|---|---|---|---|---|
| uniform_E025 | 218 000 | 482 000 | 9.37e-10 | 226 | 1683 |
| B1_E025 | 104 000 | 240 000 | 8.20e-10 | 108 | 839 |
| B3_seed0_E025 | 280 000 | 548 000 | 1.69e-10 | 302 | 1929 |
| B2_seed5_E025 | 148 000 | 316 000 | 8.85e-10 | 170 | 1098 |

Batch : `run_all.py --N 256 --tol 1e-9 --max-iter 5000000` — **5552 s** au total.

Vérification de maillage (ε/ε_uniform) : B1 1.309→1.188, B3 0.593→0.532,
B2 0.005→0.003 — **classement qualitatif robuste au maillage**.

---

## 4. Campagne sensibilité B1/A2 — référence FIXE (sweep_b1_a2.py)

**Attention** : cette campagne compare ε_B1(A2) au dénominateur **FIXE**
ε_uniform(U=0.5) — elle répond à « quand la 2e harmonique dépasse-t-elle la
référence du point opératoire ? ». Ce n'est PAS une comparaison iso-énergétique
(voir §5). Interprétation voie §5.

Série : `A2·sin(2πx)`, E = A²/2 libre, **pas de renormalisation**.

### 4.1 Scan N=128 (tol 1e-8), `results/sweep_b1_A2_128.csv`

| A2 | E | ε | ε/ε_uniform(0.5) | ε_xx | ε_yy | ε_xy | frac_xy | K | Z | P_lid | P_lid/ε | iters | résid. | t(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.45 | 0.1013 | 3.495e-3 | 0.452 | 5.19e-4 | 5.13e-4 | 2.462e-3 | 0.705 | 3.94e-3 | 1.783 | 3.386e-3 | 0.969 | 130 000 | 8.28e-9 | 140 |
| 0.55 | 0.1513 | 5.601e-3 | 0.724 | 7.48e-4 | 7.39e-4 | 4.114e-3 | 0.735 | 6.12e-3 | 2.853 | 5.401e-3 | 0.964 | 126 000 | 6.46e-9 | 127 |
| 0.60 | 0.1800 | 6.875e-3 | 0.889 | 8.77e-4 | 8.66e-4 | 5.132e-3 | 0.747 | 7.36e-3 | 3.499 | 6.617e-3 | 0.963 | 118 000 | 6.69e-9 | 127 |
| 0.64855 | 0.2103 | 8.258e-3 | 1.067 | 1.013e-3 | 9.99e-4 | 6.247e-3 | 0.756 | 8.67e-3 | 4.201 | 7.935e-3 | 0.961 | 112 000 | 5.31e-9 | 114 |
| 0.68355 | 0.2336 | 9.347e-3 | 1.208 | 1.117e-3 | 1.102e-3 | 7.128e-3 | 0.763 | 9.67e-3 | 4.753 | 8.970e-3 | 0.960 | 106 000 | 7.05e-9 | 115 |
| 0.70711 | 0.2500 | 1.0123e-2 | 1.309 | 1.190e-3 | 1.174e-3 | 7.759e-3 | 0.766 | 1.037e-2 | 5.146 | 9.708e-3 | 0.959 | 104 000 | 8.31e-9 | 105 |
| 0.75 | 0.2813 | 1.1628e-2 | 1.503 | 1.331e-3 | 1.312e-3 | 8.986e-3 | 0.773 | 1.170e-2 | 5.909 | 1.114e-2 | 0.958 | 104 000 | 7.80e-9 | 111 |
| 0.85 | 0.3613 | 1.5607e-2 | 2.017 | 1.692e-3 | 1.666e-3 | 1.225e-2 | 0.785 | 1.507e-2 | 7.924 | 1.491e-2 | 0.955 | 92 000 | 4.64e-9 | 92 |
| 1.00 | 0.5000 | 2.2856e-2 | 2.954 | 2.324e-3 | 2.286e-3 | 1.825e-2 | 0.798 | 2.082e-2 | 11.59 | 2.175e-2 | 0.952 | 90 000 | 5.26e-9 | 96 |

Croisement N=128 (ε_B1 = ε_uniform[0.5]) : **ε monotone croissant en A2**,
A2\* = **0.6302**, E\* = **0.199**.

### 4.2 Confirmation N=256 (tol 1e-9), `results/sweep_b1_A2_256.csv`

| A2 | E | ε | ε/ε_uniform(0.5) | ε_xx | ε_yy | ε_xy | frac_xy | K | Z | P_lid | P_lid/ε | iters | résid. | t(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.60 | 0.1800 | 7.620e-3 | 0.802 | 9.34e-4 | 9.30e-4 | 5.756e-3 | 0.755 | 7.66e-3 | 3.844 | 7.478e-3 | 0.981 | 252 000 | 8.68e-10 | 892 |
| 0.63024 | 0.1986 | 8.571e-3 | 0.902 | 1.023e-3 | 1.019e-3 | 6.528e-3 | 0.762 | 8.49e-3 | 4.323 | 8.407e-3 | 0.981 | 258 000 | 8.59e-10 | (batch) |
| 0.64855 | 0.2103 | 9.178e-3 | 0.966 | 1.079e-3 | 1.075e-3 | 7.023e-3 | 0.765 | 9.02e-3 | 4.628 | 9.001e-3 | 0.981 | 254 000 | 9.03e-10 | 833 |
| 0.68355 | 0.2336 | 1.0407e-2 | 1.096 | 1.191e-3 | 1.187e-3 | 8.029e-3 | 0.771 | 1.006e-2 | 5.247 | 1.020e-2 | 0.980 | 246 000 | 8.78e-10 | 766 |
| 0.70711 | 0.2500 | 1.1285e-2 | 1.188 | 1.270e-3 | 1.265e-3 | 8.750e-3 | 0.775 | 1.079e-2 | 5.689 | 1.106e-2 | 0.980 | 240 000 | 8.20e-10 | 786 |

- **Croisement N=256** : A2\* = **0.6577**, E\* = **0.2163**.
- **Marge du point opératoire** (A2=√0.5 vs croisement) : **×1.075**.
- Le croisement se déplace légèrement avec le maillage (0.199 → 0.216) — marge
  étroite robuste, mais interprétation physique limitée (référence non
  iso-énergétique).

---

## 5. Campagne energy-matched (sweep_energy_matched.py) — comparaison iso-énergétique

Principe : **même E\* = A²/2 des deux côtés**. Harmonique `A·sin(kπx)`
(k=2 : B1 ; k=1 : 1er harmonique) vs lid uniforme **renormalisé**
`U0 = A/√2 = √E\*`. Ratio = ε_contrôle / ε_uniform au même E\*. Logique de
reprise : réutilisation de `B1sweep_*`/`B1em_*` (k=2), `S{k}em_*` (k≠2),
`UNem_<U0>` / `uniform_E025` (U0=0.5) — seuls les runs manquants sont exécutés
(dispatch par `scalars_*.json`).

### 5.1 k=2 (B1) — N=256, tol 1e-9 — `results/energy_matched_256_k2.csv`

| A2 | E\* | U0 | ε_B1 | ε_uniform(√E\*) | **ratio** | B-side | U-side | temps |
|---|---|---|---|---|---|---|---|---|
| 0.60 | 0.1800 | 0.424264 | 7.620e-3 | 6.634e-3 | **1.1487** | B1sweep_0p6 | UNem_0p424264 | réutilisé+1695 s* |
| 0.63024 | 0.1986 | 0.445647 | 8.571e-3 | 7.384e-3 | **1.1606** | B1sweep_0p63024 | UNem_0p445647 | réutilisé+1695 s* |
| 0.64855 | 0.2103 | 0.458594 | 9.178e-3 | 7.861e-3 | **1.1675** | B1sweep_0p64855 | UNem_0p458594 | réutilisé+1695 s* |
| 0.65 | 0.2113 | 0.459619 | 9.227e-3 | 7.899e-3 | **1.1681** | B1em_0p65 | UNem_0p459619 | 828+1639 s |
| 0.658 | 0.2165 | 0.465276 | 9.501e-3 | 8.113e-3 | **1.1710** | B1em_0p658 | UNem_0p465276 | 871+1679 s |
| 0.666 | 0.2218 | 0.470933 | 9.779e-3 | 8.331e-3 | **1.1739** | B1em_0p666 | UNem_0p470933 | 812+1597 s |
| 0.68355 | 0.2336 | 0.483343 | 1.0407e-2 | 8.818e-3 | **1.1801** | B1sweep_0p68355 | UNem_0p483343 | réutilisé+1695 s* |
| 0.70711 | 0.2500 | 0.5 | 1.1285e-2 | 9.498e-3 | **1.1882** | B1em_0p707107 | uniform_E025 | 825 s+réutilisé |

\*temps indiqués depuis le CSV d'origine (batch 1 : 1622–1770 s selon point).

**Résultat : ratio > 1 partout, monotone croissant 1.149 → 1.188, AUCUN
croisement.** À énergie égale, la 2e harmonique B1 est **plus dissipative** que
l'uniforme sur tout [E\* = 0.18, 0.25].

### 5.2 k=1 (1er harmonique, sin πx) — N=256, tol 1e-9 — `results/energy_matched_256_k1.csv`

| A2 | E\* | U0 | ε_S1 | ε_uniform(√E\*) | **ratio** | B-side | U-side | temps |
|---|---|---|---|---|---|---|---|---|
| 0.60 | 0.1800 | 0.424264 | 4.687e-3 | 6.634e-3 | **0.7066** | S1em_0p6 | UNem_0p424264 | 1720 s+réutilisé |
| 0.658 | 0.2165 | 0.465276 | 5.819e-3 | 8.113e-3 | **0.7172** | S1em_0p658 | UNem_0p465276 | 1638 s+réutilisé |
| 0.70711 | 0.2500 | 0.5 | 6.891e-3 | 9.498e-3 | **0.7255** | S1em_0p707107 | uniform_E025 | 1586 s+réutilisé |

**Résultat : ratio < 1 partout, monotone croissant 0.707 → 0.726, AUCUN
croisement.** À énergie égale, le 1er harmonique est **moins dissipatif** que
l'uniforme (et ~63 % de moins que B1 à E\*=0.18 : 4.69e-3 vs 7.62e-3).

### 5.3 k=2 (B1) — scan N=128 (tol 1e-8, 9 points) — `results/energy_matched_128.csv`

| A2 | E\* | U0 | ε_B1 | ε_uniform | **ratio** |
|---|---|---|---|---|---|
| 0.45 | 0.1013 | 0.318198 | 3.495e-3 | 2.906e-3 | **1.2027** |
| 0.55 | 0.1513 | 0.388909 | 5.601e-3 | 4.479e-3 | **1.2503** |
| 0.60 | 0.1800 | 0.424264 | 6.875e-3 | 5.410e-3 | **1.2709** |
| 0.64855 | 0.2103 | 0.458594 | 8.258e-3 | 6.407e-3 | **1.2889** |
| 0.68355 | 0.2336 | 0.483343 | 9.347e-3 | 7.185e-3 | **1.3009** |
| 0.70711 | 0.2500 | 0.5 | 1.0123e-2 | 7.737e-3 | **1.3085** |
| 0.75 | 0.2813 | 0.530330 | 1.1628e-2 | 8.800e-3 | **1.3215** |
| 0.85 | 0.3613 | 0.601041 | 1.5607e-2 | 1.1578e-2 | **1.3480** |
| 1.00 | 0.5000 | 0.707107 | 2.2856e-2 | 1.6556e-2 | **1.3805** |

Ratios N=128 plus élevés qu'à N=256 (1.20–1.38 vs 1.15–1.19) : le surcoût
dissipatif de B1 est **surestimé à maillage grossier**, mais toujours > 1.

### 5.4 Vérification maillage N=512 (point E\*=0.25, k=2, tol 1e-9) — `results/energy_matched_512_k2.csv`

| A2 | E\* | U0 | ε_B1 | ε_uniform | **ratio** | iters (B1 / UN) | résid. (B1 / UN) | t (B1 / UN) |
|---|---|---|---|---|---|---|---|---|
| 0.70711 | 0.2500 | 0.5 | 1.1899e-2 | 1.1098e-2 | **1.0721** | 912 000 / 912 000 | 9.53e-10 / 9.53e-10 | 6365 / 12751 s |

Détail scalars N=512 (UNem_0p5) : ε_xx=1.350e-3, ε_yy=1.339e-3, ε_xy=8.409e-3,
frac_xy=0.758, K=9.479e-3, Z=5.707, P_lid/ε=0.995.

**Résultat : même signe (> 1), le ratio passe de 1.188 (N=256) à 1.072
(N=512)** — pas de croisement, **magnitude grid-dépendante**. Le point N=512 est
la seule paire de confirmation iso-énergétique ; k=1 à N=512 non vérifié.

### 5.5 Synthèse energy-matched (acquis du paper)

| | k=1 sin(πx) | k=2 sin(2πx)=B1 |
|---|---|---|
| E\* ∈ [0.18, 0.25] (N=256) | ratio 0.707–0.726 (**< 1**) | ratio 1.149–1.188 (**> 1**) |
| N=128 k=2 (E\* ∈ [0.10, 0.50]) | — | ratio 1.203–1.381 (> 1) |
| N=512 k=2 (E\* = 0.25) | — | ratio 1.072 (> 1) |
| Conclusion | plus dissipatif que l'uniforme à budget égal | moins dissipatif que l'uniforme à budget égal |

→ **En comparant à énergie égale, l'uniforme renormalisé dissipe toujours moins
que B1 (k=2) et plus que le 1er harmonique (k=1)**. Le « croisement »
A2\*=0.658 (§4) est un artefact de référence FIXE. Le récit du paper :
**branch accessibility ≠ global optimality** — la branche B1 du PINN est
accessible mais pas dissipativement optimale sous contrainte iso-énergétique.

---

## 6. Figures produites

| Figure | Contenu |
|---|---|
| `fig_lid_controls.pdf/png` | 1×4 structure U_lid (phases + moyenne pour B2/B3) |
| `fig_composite_4x3.pdf/png` | 4 contrôles × (U_lid / streamlines+vorticity / Φ log10) |
| `fig_quantitative.pdf/png` | (a) ε/ε_uniform, (b) décomposition, (c) P_lid/ε, (d) R_K |
| `fig_tau_w.pdf/png` | τ_w(x) des 4 contrôles |
| `fig_sweep_b1.pdf/png` | sensibilité B1/A2 (référence fixe) |
| `fig_energy_matched_k1_N256.{pdf,png}` | ratio k=1 vs uniform, N=256 |
| `fig_energy_matched_k2_N256.{pdf,png}` | ratio k=2 vs uniform, N=256 |
| `fig_energy_matched_k2_N512.{pdf,png}` | point N=512 + série N=256 en fond |
| `fig_summary_numbers.csv` | table des nombres-clés |

---

## 7. Commandes de reproduction

```powershell
# Production (4 contrôles, N=256) : build → run → figures → checksums
py -3.11 run_all.py

# Sensibilité B1/A2 (référence fixe)
py -3.11 sweep_b1_a2.py --N 256 --tol 1e-9 --max-iter 5000000

# Energy-matched k=2 (B1) N=256
py -3.11 sweep_energy_matched.py --N 256 --tol 1e-9 --max-iter 5000000 --k 2 --a2 0.60 0.63024 0.64855 0.650 0.658 0.666 0.68355 0.70710678

# Energy-matched k=1 (sin πx) N=256
py -3.11 sweep_energy_matched.py --N 256 --tol 1e-9 --max-iter 5000000 --k 1 --a2 0.60 0.658 0.70710678

# Energy-matched k=2 N=512 (point E*=0.25)
py -3.11 sweep_energy_matched.py --N 512 --tol 1e-9 --max-iter 5000000 --k 2 --a2 0.70710678
```

Contrôles temporels : patch solveur `stat_cycles=48` ajouté dans
`PoF_R_lid_driven_paper/lbm_mrt_validation/lbm_mrt_pinn_validation.py`
(`__init__(stat_cycles=None)` — inoffensif sans). Solveur importé depuis
`PoF_R_lid_driven_paper/lbm_mrt_validation` (chemin config.json).

---

## 8. Reproductibilité

- **Checksums SHA256 (16 hex)** des fichiers de production :
  `logs/reproduction_checksums.txt` (générés 2026-09-16 00:59:32). Points
  clés : `case_study_summary.csv` 69e6256707b6bc4a, `scalars_B1_E025_Re500_N256`
  a6edf973503d02c9, `fields_B1_E025_Re500_N256.npz` b36f18428fcc0b0b,
  `case_controls_energy.csv` cc360cf630cfcf79, séries
  `data/lid_series/{uniform_E025,B1_E025,B3_seed0_E025,B2_seed5_E025}_re500_e0.25.npz`.
- **Séries disjointes du solveur** : les contrôles PINN sont des NPZ
  (`data/lid_series/*.npz`) indépendants du LBM.
- **Dispatch de reprise** : tout script relancé ne fait que les runs manquants
  (présence de `scalars_<control>_Re500_N<N>.json`).
- Résidus finaux : 1.7e-10 à 9.5e-10 (production N=256/N=512), 4.6e-9 à 8.3e-9
  (N=128).
- Environnement : `py -3.11` requis (Python 3.11), numba, numpy, matplotlib.

---

## 9. Notes de méthode (audit)

- Φ convention : ν(2S11²+2S22²+4S12²) = 2ν S:S — idempotent à Δ=0 vs solveur.
- P_lid calculé avec la vitesse RÉCUPÉRÉE u[N−1,:] (fermeture 0.93–0.99 au lieu
  de 1.18/1.08/1.05 avec le profil idéalisé : singularité de coin u≈0.27 vs 1).
- R_K/K_fluct via `solver.compute_integrals()` (patch stat_cycles).
- Le point `A2=0.63024` du sweep N=256 a été exécuté dans un batch cumulatif
  (t 2.6e4 s intégrées dans le CSV d'origine) — valeur de ε inchangée.