# Résultats du case study — lecture physique (2026-09-15, E=0.25)

Contexte : Re=500, AR=1, **énergie d'actuation fixée E\*=⟨U_lid²⟩=0.25 pour tous
les contrôles**. Convention ν = 1/Re, champs normalisés par U_ref. Produits à
**N=256 (production, tol 1e-9)** ; N=128 (tol 1e-8) conservé pour la vérification
de maillage. Pour les contrôles temporels B3/B2, moyennes sur 48 cycles forcés
verrouillés (stat_iters=12288, patch `stat_cycles` — sans lui la convergence
précoce donnerait une phase isolée au lieu de la moyenne ; cf. config.json).

## Table des grandeurs production (N=256, tol 1e-9)

| Contrôle | type | ε | ε/ε_uniform | ε_xx | ε_yy | ε_xy | frac_xy | K | K_fluct | R_K | Z | P_lid/ε |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Uniform | stationnaire U=0.5 | 9.498e-3 | 1.000 | 1.21e-3 | 1.20e-3 | 7.09e-3 | 0.746 | 9.35e-3 | 0 | 0 | 4.92 | 0.994 |
| B1 (A2 sin 2πx, A2=√0.5) | stationnaire | 1.129e-2 | **1.188** | 1.36e-3 | 1.34e-3 | 8.58e-3 | 0.760 | 1.08e-2 | 0 | 0 | 5.69 | 0.980 |
| B3 (seed 0, spatio-temporel) | temporel (f_t=45%) | 5.051e-3 | **0.532** | 8.3e-4 | 8.2e-4 | 3.40e-3 | 0.673 | 3.86e-3 | 1.18e-4 | 0.031 | 2.56 | 0.935 |
| B2 (seed 5, pur temporel) | temporel (f_t=100%) | 3.098e-5 | **0.003** | 6.1e-6 | 6.0e-6 | 1.9e-5 | 0.622 | 1.55e-5 | 2.31e-4 | **14.96** | 0.016 | 0.365 |

(ε du champ MOYEN ; pour B2 le transfert d'énergie est porté par les fluctuations
d'où P_lid/ε=0.365 ≪ 1 — limite de la décomposition champ-moyen, cf. §4.)

Vérification de maillage N=128 → N=256 : ε/ε_uniform 1.31→1.19 (B1), 0.59→0.53
(B3), 0.005→0.003 (B2) ; P_lid/ε amélioré à 0.93–0.99 pour les cas
stationnaires/moyens. **Le classement qualitatif et le mécanisme sont robustes au
maillage.**

## Sensibilité B1/A2 — la marge d'amplitude du contrôle stationnaire est étroite

> **Important (audit 16_09_2026)** : cette section compare ε_B1(A2) au
> dénominateur **FIXE** ε_uniform(U=0.5) (référence E*=0.25). Elle répond à
> « à quelle amplitude la 2e harmonique dépasse-t-elle la référence du point de
> fonctionnement ? ». Ce N'EST PAS une comparaison iso-énergétique — elle ne
> doit PAS être interprétée comme « B1 vs uniforme à même E* ». La comparaison
> iso-énergétique stricte est dans la section suivante (`## Comparison
> iso-énergétique`), seule valable pour le narratif « branch accessibility ≠
> optimality ».

Sweep `A2` de `A2·sin(2πx)` (E=A²/2 libre, N=256, tol 1e-9, 5 points) ;
`results/sweep_b1_A2_256.csv` + `figures/fig_sweep_b1.pdf`.

| A2 | E=A²/2 | ε | ε/ε_uniform(0.5) | Z |
|---|---|---|---|---|
| 0.600 | 0.180 | 7.620e-3 | 0.802 | 3.84 |
| 0.63024 | 0.199 | 8.571e-3 | 0.902 | 4.32 |
| 0.64855 | 0.210 | 9.178e-3 | 0.966 | 4.63 |
| 0.68355 | 0.234 | 1.041e-2 | 1.096 | 5.25 |
| √0.5 = 0.70711 | 0.250 | 1.129e-2 | 1.188 | 5.69 |

- **Croisement (référence fixe)** : **A2\* = 0.658 (E\*=0.2163)** — ε est
  monotone croissant en A2 (vérifié sur les 2 trames).
- Le point de fonctionnement du paper (A2=√0.5, E=0.25) n'est qu'à **+7.5 %
  d'amplitude** au-dessus du basculement (marge ×1.075).
- Maillage : le croisement se déplace (N=128 : A2\*=0.630, E\*=0.199) — valeur
  chiffrée à prendre à N=256.

## Comparison iso-énergétique (energy-matched) — B1 vs uniform vs 1er harmonique

Même **énergie d'actuation E\* = ⟨U_lid²⟩ = A²/2** de part et d'autre : la 2e
harmonique `A·sin(2πx)` (k=2) ou le 1er harmonique `A·sin(πx)` (k=1) contre le
lid uniforme **renormalisé** `U0 = A/√2 = √E*`. Ratio = ε_contrôle/ε_uniform
au **même E\***. Script `code/sweep_energy_matched.py`, sorties
`results/energy_matched_<N>_k<k>.csv` + `figures/fig_energy_matched_k<k>_N<N>.pdf/png`.

**k=2 (B1) — N=256, tol 1e-9, résidus ~8e-10 :**

| A2 | E* | ε_A·sin(2πx) | ε_uniform(√E*) | ratio |
|---|---|---|---|---|
| 0.600 | 0.180 | 7.620e-3 | 6.634e-3 | **1.1487** |
| 0.63024 | 0.1986 | 8.571e-3 | 7.384e-3 | **1.1606** |
| 0.64855 | 0.2103 | 9.178e-3 | 7.861e-3 | **1.1675** |
| 0.650 | 0.2113 | 9.227e-3 | 7.899e-3 | **1.1681** |
| 0.658 | 0.2165 | 9.501e-3 | 8.113e-3 | **1.1710** |
| 0.666 | 0.2218 | 9.779e-3 | 8.331e-3 | **1.1739** |
| 0.68355 | 0.2336 | 1.0407e-2 | 8.819e-3 | **1.1801** |
| 0.70711 | 0.250 | 1.1285e-2 | 9.498e-3 | **1.1882** |

**k=1 (sin πx) — N=256, tol 1e-9 :**

| A2 | E* | ε_A·sin(πx) | ε_uniform(√E*) | ratio |
|---|---|---|---|---|
| 0.60 | 0.180 | 4.687e-3 | 6.634e-3 | **0.7066** |
| 0.658 | 0.2165 | 5.819e-3 | 8.113e-3 | **0.7172** |
| 0.70711 | 0.250 | 6.891e-3 | 9.498e-3 | **0.7255** |

**Arrêté des résultats iso-énergétiques :**

1. **AUCUN croisement à N=256** : k=2 (B1) est toujours **plus dissipatif** que
   l'uniforme au même E\* (ratio 1.149 → 1.188, monotonie croissante), tandis que
   k=1 (1er harmonique) est toujours **moins dissipatif** (ratio 0.707 → 0.726,
   monotonie croissante). L'uniforme se situe entre les deux.
2. **Le « croisement » A2\*=0.658 de la section précédente est un artefact de
   référence FIXE** : à énergie égale, B1 ne « devient » jamais mieux que
   l'uniforme dans [0.18, 0.25]. Le gain apparent du PINN dans la campagne
   E*≤0.25 pour la branche B1 est donc une **baisse d'amplitude**, pas une
   supériorité structurale iso-énergétique → **branch accessibility ≠ global
   optimality** (le récit du paper).
3. **Robustesse au maillage (réservée)** : au point du paper (E*=0.25, k=2),
   le ratio passe de **1.188 (N=256)** à **1.072 (N=512)** — même signe > 1,
   pas de croisement, mais **magnitude grid-dépendante** (N=128 : ~1.20–1.38 sur
   [0.10, 0.50]). Le point N=512 reste le seul point de confirmation iso-énergétique
   (1 paire) ; le k=1 à N=512 n'a PAS été vérifié.
4. **Structure spatiale dominante** : à E* identique, le 1er harmonique dissipe
   ~63 % de moins que le 2e (E*=0.18 : 4.69e-3 vs 7.62e-3). Le « portage » du
   contrôle se joue sur f_2 (harmonique) plus que sur la classe temporelle pure.

## Lecture mécanistique — à budget d'actuation fixé, la STRUCTURE du contrôle domine

1. **Le contrôle purement stationnaire à 2e harmonique (B1) DÉGRADE le bilan
   d'énergie** par rapport au lid plat : ε/ε_uniform = **1.19**. À E\* constant,
   redistribuer la même énergie sur une onde stationnaire (x) excite des gradients
   de strain supplémentaires (S_xy, frac_xy=0.76) sans gain de dissipation. Le
   gain apparent du PINN dans la campagne E\*≤0.25 était donc porté par la
   **baisse d'amplitude**, pas par la structure spatiale seule.

2. **B3 (mélange B2+B3 du PINN) réduit de ~47 %** (ε/ε_uniform=0.53) tout en
   restant quasi-stationnaire dans la réponse (R_K=0.031) : la composante
   temporelle du contrôle agit comme un « amortisseur » modal (Z dégonflé ~52 %)
   sans oscillation du champ moyen.

3. **B2 (pur temporel) effondre la dissipation du champ moyen à 0.3 % du lid
   plat** (ε/ε_uniform=0.003), mais le régime N'est PAS stationnaire :
   R_K=14.96 ⇒ l'énergie cinétique des fluctuations du cycle forcé dépasse ~15×
   celle du champ moyen. Là, ε(⟨u⟩) sous-estime la dissipation instantanée réelle
   (P_lid/ε=0.37) : limite documentée de la décomposition champ-moyen, et motif
   justifiant R_K dans les diagnostics.

4. **Cisaillement (ε_xy) dominant mais réduit** : frac_xy 0.75 (Uniform) → 0.62
   (B2) : les contrôles « lissent » progressivement la singularité de coin du lid.

5. **Z suit grossièrement ε** (4.92 → 2.56 → 0.016) : réduction de population de
   vorticité colinéaire à ε, pas de re-localisation spectaculaire de Φ.

6. **Bilan P_lid≈ε vérifié à ~1–6 %** sur les cas stationnaires/moyens
   (0.93–0.99). Écart B2 (0.37) = témoin du transfert par les fluctuations.

## Notes de méthode (audit 15_09_2026)
- Φ convention : ν(2S11²+2S22²+4S12²) = 2ν S:S — idempotent à Δ=0 vs solveur.
- P_lid calculé avec la vitesse RÉCUPÉRÉE u[N-1,:] (fermeture 0.93–0.99 au lieu
  de 1.18/1.08/1.05 avec le profil idéalisé : singularité de coin u≈0.27 vs 1).
- R_K/K_fluct via solver.compute_integrals() (patch stat_cycles).
- Exécution batch N=256 : run_all --N 256 --tol 1e-9 --max-iter 5e6, 5552 s au
  total ; résidus 8.2e-10 à 1.7e-10 ; checksums logs/reproduction_checksums.txt.

## Prochaines étapes
- Alimenter la Discussion Paper 2 (items 1–3 : branches B1/B3/B2 à budget fixé) +
  Encadré « Énergie d'actuation » (E=0.25 identique, α=1.01–1.02).
- Comparaison τ_w(x) N=256 (fig_tau_w) pour annotation manuscrit.
- (Option) étude de sensibilité A2/B1 autour de 0.25 pour cerner la valeur de
  croisement B1≈uniform.