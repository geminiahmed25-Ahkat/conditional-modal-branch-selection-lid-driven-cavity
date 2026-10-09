# Notice to co-authors - taxonomy correction in Paper 2

**Subject:** Correction of the modal-branch taxonomy (`B_first`) in the Paper 2
manuscript - self-disclosure before/at revision.

---

## Version francaise

Chers coauteurs,

En preparant un paquet de reproduction propre (code + donnees + figures + LaTeX
verifies) du manuscrit *"Conditional modal branch selection in physics-informed
active control of lid-driven cavity flow"*, j'ai identifie une erreur de
classification dans notre taxonomie des branches. Je la signale par transparence
et je propose de la corriger dans la version revisee.

**Ce qui etait inexact**

1. Notre schema a trois classes (`B1` dominant mode 2, `B2` domine par le temps,
   `B3` mixte) rangeait dans `B3` toutes les realisations qui n'etaient ni `B1`
   ni `B2`. En realite, les runs obtenus avec **regularisation de variance nulle
   (`lambda_var = 0`)** ne sont pas mixtes : ils sont domines par le **premier
   harmonique stationnaire** (fraction `f_1 = Efrac_0_0 > 0.9`). Il s'agit d'un
   etat physique distinct, celle d'un controle `sin(pi x)`, qui doit former une
   classe a part : `B_first`.
2. Le texte principal annonçait des seuils `f_2 > 0.70` et `f_t > 0.70`. La
   regle reellement implementee (et utilisee pour toutes les etiquettes) est la
   regle de dominance `f_2 > 2 f_t et f_2 > 0.5` (resp. `f_t > 2 f_2 et
   f_t > 0.5`). Les deux regles donnent les **memes** etiquettes sur la grille de
   reference a 185 runs, mais le texte ne decrivait pas la regle utilisee.
3. L'entropie de Shannon normalisee etait divisee par `ln 3`; avec quatre
   classes elle doit etre divisee par `ln 4`.

**Portee**

- La campagne de reference a 185 runs (`lambda_var = 1`) ne contient **aucune**
  realisation `B_first` (`f_1 <= 0.13` partout). Ses statistiques publiees
  (`B_1` 42.2 %, `B_2` 43.8 %, `B_3` 14.1 %; 0/185 divergences) sont **inchangees**,
  et l'ensemble des conclusions principales tient.
- La correction modifie l'interpretation du **balayage `lambda_var`** et de
  l'**ablation de perte** : a `lambda_var = 0` le modele selectionne de facon
  deterministe `B_first` (et non un controle "mixte" desordonne); toute valeur
  positive de `lambda_var` supprime `B_first`. Le resultat "No L_var" est
  desormais `B_1 -> B_first` et non `B_1 -> mixte`.
- Elle renforce le message de l'article : la penalite de variance **exclut** le
  mode fondamental de plus faible dissipation.

**Ce qui a ete prepare**

Un paquet corrige et entierement reproductible (`Paper2_clean_corrected`):
taxonomie a quatre classes (`branch_labels.py`), tables recalculees, figures 3/4/7
regenerees, manuscrit et supplement LaTeX corriges (PDF compiles), rapport de
correction et registre de reutilisation des donnees Paper 3
(`05_correction_report/`).

**Ce que je propose**

Soumettre a l'editeur la version corrigee du manuscrit et du supplement (fichiers
joints), accompagnes d'une note de transparence. Merci de me confirmer votre
accord et de me signaler toute remarque avant envoi.

Bien cordialement,
[Nom]

---

## English version

Dear co-authors,

While assembling a clean, fully reproducible package (code + data + figures +
verified LaTeX) for the manuscript *"Conditional modal branch selection in
physics-informed active control of lid-driven cavity flow"*, I found a
classification error in our branch taxonomy. I am disclosing it for transparency
and proposing to fix it in the revised version.

**What was inaccurate**

1. Our three-class scheme (`B1` mode-2 dominated, `B2` time-dominated, `B3` mixed)
   placed every realization that was neither `B1` nor `B2` into `B3`. In fact the
   runs obtained with **zero variance regularization (`lambda_var = 0`)** are not
   mixed: they are dominated by the **stationary first harmonic**
   (`f_1 = Efrac_0_0 > 0.9`). This is a distinct physical state (a `sin(pi x)`
   control) and must form its own class: `B_first`.
2. The main text quoted thresholds `f_2 > 0.70` and `f_t > 0.70`, whereas the
   rule actually implemented (and used for every label) is the dominance rule
   `f_2 > 2 f_t and f_2 > 0.5` (resp. `f_t > 2 f_2 and f_t > 0.5`). The two rules
   give **identical** labels on the 185-run reference grid, but the text did not
   describe the rule that was used.
3. The normalized Shannon entropy was divided by `ln 3`; with four classes it must
   be divided by `ln 4`.

**Impact**

- The 185-run reference campaign (`lambda_var = 1`) contains **no** `B_first`
  realization (`f_1 <= 0.13` throughout). Its published statistics
  (`B_1` 42.2 %, `B_2` 43.8 %, `B_3` 14.1 %; 0/185 discrepancies) are
  **unchanged**, and all main conclusions hold.
- The correction changes the interpretation of the **`lambda_var` sweep** and the
  **loss ablation**: at `lambda_var = 0` the model deterministically selects
  `B_first` (not a disordered "mixed" control); any positive `lambda_var`
  suppresses `B_first`. The "No L_var" result becomes `B_1 -> B_first` instead of
  `B_1 -> mixed`.
- It strengthens the paper's message: the variance penalty **excludes** the
  lowest-dissipation fundamental mode.

**What has been prepared**

A corrected, fully reproducible package (`Paper2_clean_corrected`): four-class
taxonomy (`branch_labels.py`), recomputed tables, regenerated figures 3/4/7,
corrected manuscript and supplementary LaTeX (compiled PDFs), a correction report,
and a Paper 3 data-reuse registry (`05_correction_report/`).

**Proposed action**

Submit the corrected manuscript and supplementary to the editor (attached),
together with a transparency note. Please confirm your agreement and send any
remarks before I send it.

Best regards,
[Name]
