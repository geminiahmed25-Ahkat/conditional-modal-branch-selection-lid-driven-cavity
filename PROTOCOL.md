# Protocole détaillé — Case study mécanistique Re=500, E\*=0.25 (2026-09-15)

Point d'opération : Re=500, AR=1, **énergie d'actuation fixée E\* = ⟨U_lid²⟩ =
0.25 pour TOUS les contrôles** (la structure seule varie). Question du case study :
à budget d'actuation identique, comment l'organisation modale du contrôle
(uniforme / stationnaire 2e harmonique B1 / mixte B3 / temporel pur B2)
réorganise-t-elle circulation, taux de strain, cisaillement pariétal et
dissipation ?

## 1. Construction des contrôles (code/build_controls.py)

Énergie ⟨U²⟩ pondérée par trapèzes sur la grille (x ∈ [0,1], t ∈ [0,2] pour les
temporels) ; ingestion dans le LBM brute (amplitude physique U_ref·U_lid_norm).

| Contrôle | Fichier NPZ | Construction | E mesuré | α |
|---|---|---|---|---|
| uniform_E025 | uniform_E025_re500_e0.25 | U(x)=√0.25 = 0.5 | 0.2500 | — |
| B1_E025 | B1_E025_re500_e0.25 | A2·sin(2πx), A2=√0.5=0.70710678 | 0.2500 | — |
| B3_seed0_E025 | B3_seed0_E025_re500_e0.25 | série complète seed 0, α=√(0.25/⟨U²⟩_x,t)=1.02109 | 0.2500 | 1.0211 |
| B2_seed5_E025 | B2_seed5_E025_re500_e0.25 | série complète seed 5, α=1.00956 | 0.2500 | 1.0096 |

Séries PINN : seeds 0..14 de `results/01_ReE_star_map/re500_e0.25/seed_*/model.pt`
(`UltraPINN(basis='fourier')`, x 256 pts, t ∈ [0,2] en 256 pas) ; CSV
`seed_controls_re500_e0.25.csv` (A_k0, E_x_t par seed) + `case_controls_energy.csv`
(budget vs mesuré, α, A2, provenance f_2/f_t de la campagne).

**Expérience A (archive _v0_archive, non incluse dans le case study E fixé) :
uniform(amplitude 1), sin_2pi (A2=0.68355, L208), PINN(mean) — validation LBM du
contrôle appris (reste utilisable seul).**

## 2. Résolution LBM-MRT (code/run_case_study.py, code/diagnostics.py)

- Solveur `LBM_MRT_Solver` (D2Q9-MRT, U_ref=0.05), lid via `U_lid2d[phase,j]`,
  période = nt pour B2/B3 (échantillonnage t 256 pts / 2.0), période = 1 sinon.
- N=128 (validation, tol 1e-8, convergence : erreur de périodicité max sur
  sonde, 3 contrôles consécutifs sous tol, 2 000 pas entre checks).
- **Moyenne temporelle** : accumulateurs internes (2e moitié de la fenêtre max_iter).
  Pour les contrôles temporels convergés AVANT la fenêtre (settle_at = max_iter//2),
  on applique le patch solveur `stat_cycles=48` : re-moyenne sur 48 cycles forcés
  verrouillés → µ vrais (stat_iters=48·256=12288), sinon le « champ moyen » serait
  une phase isolée (dépouillé : R_K=0). Config : `lbm.stat_cycles`.
- Diagnostics : Φ, ε_xx/ε_yy/ε_xy (ν=1/Re, champs /U_ref), K, K_fluct, Z,
  **R_K = K_fluct/K** (essentialité de l'instationnarité pour B2/B3),
  ψ (∫₀ᵧu ds), τ_w(x)=ν·(u_{N-1}−u_{N-2})/dx, **P_lid = −∫ U_lid(récupéré)·τ_w dx**
  (fermeture ~1 % à N=128).

## 3. Bilan d'énergie attendu

P_lid ≈ ε (cas stationnaires/moyens ; |P_lid/ε−1| < 5 % à N=128, < 3 % à N=256).
**Cas temporel pur (B2)** : le champ moyen perd sa validité (R_K ≫ 1, P_lid/ε
≪ 1) → le bilan doit être discuté sur les moments d'ordre 2 (K_fluct) ; c'est une
limite documentée de la décomposition champ-moyen, et le motif physique principal
du case study.

## 4. Figures (code/make_figures.py) — labels EN ANGLAIS (exigence utilisateur)

- `fig_lid_controls` : 1×4 structure U_lid (phases + moyenne pour B2/B3).
- `fig_composite_4x3` : 4 colonnes (Uniform, B1, B3, B2) × 3 lignes
  (U_lid / streamlines+vorticity / Φ(x,y) log10).
- `fig_quantitative` : (a) ε/ε_uniform, (b) ε_xx/ε_yy/ε_xy,
  (c) P_lid/ε, (d) R_K (échelle symlog).
- `fig_tau_w` : τ_w(x) des 4 contrôles + `fig_summary_numbers.csv`.

## 5. Sensibilité B1/A2 (code/sweep_b1_a2.py) — marge d'amplitude (référence FIXE)

> **Référence FIXE ε_uniform(U=0.5)** (E*=0.25) pour TOUS les A2. Réponse :
> « à quelle amplitude la 2e harmonique dépasse-t-elle la référence du point de
> fonctionnement ? ». **Ce n'est PAS une comparaison iso-énergétique** — la
> comparaison à énergie égale est traitée en §5 bis (energy-matched), seule
> valable pour le narratif du paper.

L'amplitude A2 de `A2·sin(2πx)` est le paramètre libre (E = A²/2, PAS renormalisé) ;
on résout chaque point au LBM et on cherche la **valeur de croisement A2\*** telle
que ε_B1(A2\*) = ε_uniform(U=0.5). Points : [0.45, 0.55, 0.60, 0.64855, 0.68355,
√0.5, 0.75, 0.85, 1.00] à N=128 (scan), puis [0.60, 0.63024, 0.64855, 0.68355,
√0.5] à N=256 (confirmation). Séparateurs : série stationnaire B1sweep_a2_<v> ; sorties
`results/sweep_b1_A2_<N>.csv` + `figures/fig_sweep_b1.pdf/png`. Interpolation
linéaire de ε(A2) (ε monotone croissant en A2, vérifié sur les 2 trames).

**Résultats (N=256, tol 1e-9, résidus ~8–9e-10)** : A2\* = **0.6577**
(E\* = **0.2163**) ; marge du point de fonctionnement √0.5 : **×1.075** (seulement
+7.5 % d'amplitude au-dessus du seuil de bascule). A N=128 : A2\*=0.6302 (E*=0.199)
— le croisement se déplace avec le maillage (0.199 → 0.216), mais la marge reste
étroite dans les deux cas. Ratios N=256 : 0.60→0.802, 0.63024→0.902, 0.64855→0.966,
0.68355→1.096, √0.5→1.188 (reproductibilité exacte de B1_E025 production ✓).

## 5 bis. Comparison iso-énergétique (code/sweep_energy_matched.py) — energy-matched

Même **énergie d'actuation E\* = ⟨U_lid²⟩ = A²/2** des deux côtés : la 2e
harmonique `A·sin(2πx)` (k=2) **ou** le 1er harmonique `A·sin(πx)` (k=1) contre
le lid uniforme **renormalisé** `U0 = A/√2 = √E*`. Ratio = ε_contrôle/ε_uniform
au MÊME E\*. Le script réutilise les séries B1sweep_*/B1em_* (k=2) ou crée
S{k}em_* (k≠2), et les uniforms energy-matched UNem_<U0> (uniform_E025 quand
U0=0.5) ; seuls les runs manquants sont lancés (reprise par scalars_*.json).

Usage :
```
py -3.11 sweep_energy_matched.py --N 256 --tol 1e-9 --max-iter 5000000 --k 2 --a2 0.60 0.658 0.70711
py -3.11 sweep_energy_matched.py --N 256 --tol 1e-9 --max-iter 5000000 --k 1 --a2 0.60 0.658 0.70711
py -3.11 sweep_energy_matched.py --N 512 --tol 1e-9 --max-iter 5000000 --k 2 --a2 0.70710678 [--verbose]
```
Sorties : `results/energy_matched_<N>_k<k>.csv`,
`figures/fig_energy_matched_k<k>_N<N>.pdf/png`, logs `sweep_energy_matched_<N>_k<k>.log`.

**Résultats (N=256, tol 1e-9)** :
- **k=2 (B1) : AUCUN croisement, ratio > 1 partout** — 1.1487 (E*=0.18) →
  1.1882 (E*=0.25), monotone croissant ; B1 est **plus dissipatif** que
  l'uniforme au même budget.
- **k=1 (sin πx) : ratio < 1 partout** — 0.7066 (E*=0.18) → 0.7255 (E*=0.25),
  monotone croissant ; le 1er harmonique est **moins dissipatif** que l'uniforme
  au même budget (et ~63 % de moins que B1 à E*=0.18).
- **Vérif. maillage N=512 (point paper E*=0.25, k=2)** : ratio **1.0721**
  (B1 1.1899e-2, UN 1.1098e-2) — même signe > 1, pas de croisement, mais
  **magnitude grid-dépendante** (rappel N=128 : 1.20–1.38 sur [0.10, 0.50]).
  Point N=512 unique (1 paire) ; k=1 à N=512 NON vérifié.
- **Interprétation** : le « croisement A2\*=0.658 » du §5 est un artefact de
  **référence fixe**. À énergie égale, la 2e harmonique sélectionnée par la
  branche PINN n'est jamais dissipativement meilleure que l'uniforme :
  **branch accessibility ≠ global optimality**.

## 6. Reproductibilité

- `run_all.py` : build → run → figures → checksums SHA256
  (`logs/reproduction_checksums.txt`).
- `py -3.11` requis ; solveur importé depuis PoF_R_lid_driven_paper (chemin
  config) ; séries PINN indépendantes du solveur (NPZ). Patch `stat_cycles` dans
  `PoF_R_lid_driven_paper/lbm_mrt_validation/lbm_mrt_pinn_validation.py`
  (paramètre `__init__(stat_cycles=None)` — inoffensif par défaut).

## 7. Sorties ≥ manuscrit

Tables : (a) contrôles & amplitudes/E·α ; (b) décomposition ε_xx/ε_yy/ε_xy ;
(c) K, K_fluct, R_K, Z, P_lid, P_lid/ε ; (d) ε/ε_uniform par classe de contrôle.
Section proposée : « Focused fixed-energy comparison of the physical consequences
of competing control classes » — différentes branches de contrôle réorganisent le
transfert d'énergie et les champs de strain différemment à budget d'actuation égal.