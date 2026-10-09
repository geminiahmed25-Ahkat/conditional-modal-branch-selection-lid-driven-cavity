# -*- coding: utf-8 -*-
"""Paper 2 - shared branch feature/label definitions (single source of truth).

CORRECTED four-class taxonomy (v2, clean package).

z = (E_r, f_1, f_2, f_t, A_2, C_3, eps)

behavior rule (mirrored from the landmark separability check), extended so that
the stationary first-harmonic branch is classified explicitly instead of being
folded into the mixed class:

    B_first       f_1 > 0.5
    B1_Mode2      f_2 > 2*f_t and f_2 > 0.5
    B2_temporal   f_t > 2*f_2 and f_t > 0.5
    B3_mixed      otherwise

f_1 = Efrac_0_0 = stationary first-harmonic (fundamental) control-energy fraction.
The original three-class rule (without B_first) is retained in `behavior_name`
for backward compatibility; use `behavior_name4` / `features_and_label` for the
corrected classification. C_3 = cumulative share of the top-3 Efrac entries.
"""

import numpy as np

FEAT = ["E_r", "f_2", "f_t", "A_2", "C_3", "eps"]
FEAT4 = ["E_r", "f_1", "f_2", "f_t", "A_2", "C_3", "eps"]

BRANCH_NAME = {0: "NA", 1: "B1_Mode2", 2: "B2_temporal", 3: "B3_mixed", 4: "B_first"}
BRANCH_ID = {"NA": 0, "B1_Mode2": 1, "B2_temporal": 2, "B3_mixed": 3, "B_first": 4}


def behavior_name(f2, ft):
    """Original three-class label (no B_first). Kept for backward compatibility."""
    if f2 > 2.0 * ft and f2 > 0.5:
        return "B1_Mode2"
    if ft > 2.0 * f2 and ft > 0.5:
        return "B2_temporal"
    return "B3_mixed"


def behavior_name4(f1, f2, ft):
    """Corrected four-class label. f1 = stationary first-harmonic fraction."""
    if f1 > 0.5:
        return "B_first"
    if f2 > 2.0 * ft and f2 > 0.5:
        return "B1_Mode2"
    if ft > 2.0 * f2 and ft > 0.5:
        return "B2_temporal"
    return "B3_mixed"


def rank_concentration(efrac_values):
    return float(np.sum(np.sort(np.asarray(efrac_values, dtype=float))[::-1][:3]))


def features_and_label(ac):
    """Given one analyze_control() row (dict), return derived features + labels.

    ac must contain: mode2_fraction, temporal_fraction, A20, reconstruction_rmse,
    energy_final (or E_total), and the Energy-fraction columns Efrac_* (including
    Efrac_0_0 = first harmonic).
    """
    f2 = float(ac["mode2_fraction"])
    ft = float(ac["temporal_fraction"])
    f1 = float(ac.get("Efrac_0_0", ac.get("f_1", float("nan"))))
    ecols = sorted([k for k in ac if k.startswith("Efrac_")])
    rk = rank_concentration([ac[k] for k in ecols]) if ecols else float("nan")
    name = behavior_name4(f1, f2, ft) if np.isfinite(f1) else behavior_name(f2, ft)
    feat = {
        "E_r": float(ac.get("energy_final", ac.get("E_total", float("nan")))),
        "f_1": f1,
        "f_2": f2,
        "f_t": ft,
        "A_2": float(ac["A20"]),
        "C_3": rk,
        "eps": float(ac.get("reconstruction_rmse", float("nan"))),
    }
    return feat, name, BRANCH_ID[name]


def point_flags(counts_b1, counts_b2, counts_b3, n_na, counts_bfirst=0):
    """Adaptive-refinement decision at point level (generalized to 4 classes).

    needs_refinement: any branch probability inside (0.2, 0.8) or an NA realization.
    escalation_to_10 : force N_seed=10 at such points.
    """
    n = counts_b1 + counts_b2 + counts_b3 + counts_bfirst + n_na
    ps = [c / n for c in (counts_b1, counts_b2, counts_b3, counts_bfirst)] if n else []
    boundary = any(0.2 < p < 0.8 for p in ps) or n_na > 0
    return {"needs_refinement": bool(boundary), "escalation_to_10": bool(boundary)}
