# -*- coding: utf-8 -*-
"""
Paper 2 -- corrected taxonomy (adds B_first) and 4-class entropy.

Single source of truth for the corrected manuscript/figure numbers.

Rule (implemented exactly, consistent with branch_labels.py):
    B_first : f1 > 0.5                      (stationary first spatial harmonic, (m,n)=(1,0))
    B1      : f2 > 2*ft and f2 > 0.5        (stationary second harmonic, (m,n)=(2,0))
    B2      : ft > 2*f2 and ft > 0.5        (time-dependent actuation)
    B3      : otherwise                      (mixed)
where
    f1 = Efrac_0_0 / E_total   (fundamental / first harmonic, sin(pi x))
    f2 = mode2 / Efrac_1_0     (second harmonic, sin(2 pi x))
    ft = temporal_fraction     (sum over n>=1)
Normalized Shannon entropy over the four classes:
    H_N = -(1/ln 4) sum_k P_k ln P_k   (0 = deterministic, 1 = uniform over 4 classes)

Inputs (this folder ./inputs/):
    global_reclassification.csv   frozen per-run audit (f1/f2/ft per source run)
    loss_ablation_results.csv     frozen 8-run loss ablation (No Lvar, etc.)
    mode_analysis.csv             frozen main-grid modal analysis (seed study)
    runs_campaign2.csv            frozen 49-run Paper 3 campaign2 list (C2);
                                  only for the reuse-registry C2 summary row

Outputs (written to results/):
    corrected_taxonomy_all_runs.csv
    corrected_maingrid_summary.csv
    corrected_lambda_table.csv
    corrected_lambda_table_pooled.csv
    corrected_exp09_summary.csv
    corrected_ablation_table.csv
    corrected_seed_study.csv
    corrected_energy_sweep.csv
    corrected_global_summary.csv

Self-contained: all paths resolve relative to this script, so the correction
reproduces from a fresh clone without editing. Set P2CORR_HOME to redirect
inputs/test/outputs elsewhere (e.g. to regenerate a full package copy).
"""
import os
import numpy as np
import pandas as pd

BASE = os.getenv("P2CORR_HOME") or os.path.dirname(os.path.abspath(__file__))
INPUTS = os.path.join(BASE, "inputs")
OUT = os.path.join(BASE, "results")
os.makedirs(OUT, exist_ok=True)

AUDIT = os.path.join(INPUTS, "global_reclassification.csv")
ABLATION = os.path.join(INPUTS, "loss_ablation_results.csv")
MAINGRID_MODE = os.path.join(INPUTS, "mode_analysis.csv")
C2_RUNS = os.path.join(INPUTS, "runs_campaign2.csv")

CLASSES = ["B1", "B2", "B_first", "B3"]


def classify(f1, f2, ft):
    if f1 > 0.5:
        return "B_first"
    if f2 > 2.0 * ft and f2 > 0.5:
        return "B1"
    if ft > 2.0 * f2 and ft > 0.5:
        return "B2"
    return "B3"


def entropy4(counts):
    p = np.asarray(counts, dtype=float)
    n = p.sum()
    if n == 0:
        return np.nan
    p = p / n
    pp = p[p > 0]
    return float(-np.sum(pp * np.log(pp)) / np.log(4.0))


def summarize(df):
    c = {k: int((df["corr_label"] == k).sum()) for k in CLASSES}
    n = len(df)
    return {"N": n, **{f"n_{k}": c[k] for k in CLASSES},
            **{f"P_{k}": (c[k] / n if n else np.nan) for k in CLASSES},
            "H_N": entropy4([c[k] for k in CLASSES])}


# ---------------------------------------------------------------- audit master
g = pd.read_csv(AUDIT)
for col in ["f1_firstHarm", "f2_mode2", "ft_temporal", "E_r", "lambda_var", "Re", "E_star"]:
    g[col] = pd.to_numeric(g[col], errors="coerce")
g["corr_label"] = [classify(a, b, c) for a, b, c in zip(g.f1_firstHarm, g.f2_mode2, g.ft_temporal)]
# dominant class label for tables
def dominant(row):
    counts = {k: int((row["corr_label"] == k).sum()) for k in CLASSES}
    mx = max(counts.values())
    ties = [k for k, v in counts.items() if v == mx]
    return "/".join(ties)
g.to_csv(os.path.join(OUT, "corrected_taxonomy_all_runs.csv"), index=False)
print("Sources:", g["source"].value_counts().to_dict())

# ---------------------------------------------------------------- main grid 185
# Exclude the stray, misplaced duplicate results/01_ReE_star_map/seed_0
# (absent from the frozen 185-run CSV; see correction report).
stray = g[(g.source == "P2_maingrid") & (~g.rel.astype(str).str.contains(r"\\", na=False))]
print("\nSTRAY rows excluded from main grid:", stray[["rel", "point_id", "seed"]].to_dict("records"))
g = g.drop(stray.index)

mg = g[g.source == "P2_maingrid"]
mgsum = summarize(mg)
mgsum_tab = pd.DataFrame([{"campaign": "main grid 185 (lambda_var=1)", **mgsum}])
mgsum_tab.to_csv(os.path.join(OUT, "corrected_maingrid_summary.csv"), index=False)
print("\n=== MAIN GRID 185 ===")
print(mgsum)
# per-class modal stats
for k in CLASSES:
    s = mg[mg.corr_label == k]
    if len(s):
        print(f"  {k:7s} n={len(s):3d}  mean f1={s.f1_firstHarm.mean()*100:5.1f} "
              f"mean f2={s.f2_mode2.mean()*100:5.1f}  mean ft={s.ft_temporal.mean()*100:5.1f}")

# ---------------------------------------------------------------- lambda table
lam = g[g.source == "P2_lambdavar"].copy()
rows = []
for (re_, e_, lv), sub in lam.groupby(["Re", "E_star", "lambda_var"]):
    s = summarize(sub)
    dom = dict()
    for k in CLASSES:
        dom[k] = s[f"P_{k}"]
    mx = max(dom.values())
    ties = [k for k in CLASSES if abs(dom[k] - mx) < 1e-9 and mx > 0]
    rows.append({"Re": int(re_), "E_star": e_, "lambda_var": lv, **s, "dominant": "/".join(ties)})
lam_tab = pd.DataFrame(rows).sort_values(["Re", "E_star", "lambda_var"])
lam_tab.to_csv(os.path.join(OUT, "corrected_lambda_table.csv"), index=False)
print("\n=== LAMBDA TABLE (per cell) ===")
print(lam_tab.to_string(index=False))

# pooled over the 3 cells
pool = []
for lv, sub in lam.groupby("lambda_var"):
    s = summarize(sub)
    mx = max(s[f"P_{k}"] for k in CLASSES)
    ties = [k for k in CLASSES if abs(s[f"P_{k}"] - mx) < 1e-9 and mx > 0]
    pool.append({"lambda_var": lv, **s, "dominant": "/".join(ties)})
pool_tab = pd.DataFrame(pool).sort_values("lambda_var")
pool_tab.to_csv(os.path.join(OUT, "corrected_lambda_table_pooled.csv"), index=False)
print("\n=== LAMBDA POOLED (3 cells x 5 seeds) ===")
print(pool_tab.to_string(index=False))

# ---------------------------------------------------------------- exp09
e09 = g[g.source == "P2_09"]
rows = []
for exp, sub in e09.groupby("exp"):
    s = summarize(sub)
    mx = max(s[f"P_{k}"] for k in CLASSES)
    ties = [k for k in CLASSES if abs(s[f"P_{k}"] - mx) < 1e-9 and mx > 0]
    rows.append({"exp": exp, **s, "dominant": "/".join(ties)})
e09_tab = pd.DataFrame(rows)
e09_tab.to_csv(os.path.join(OUT, "corrected_exp09_summary.csv"), index=False)
print("\n=== EXP09 (conditional accessibility) ===")
print(e09_tab.to_string(index=False))

# ---------------------------------------------------------------- loss ablation
ab = pd.read_csv(ABLATION)
ab["f1"] = ab["Efrac_0_0"]
ab["f2"] = ab["mode2_fraction"]
ab["ft"] = ab["temporal_fraction"]
ab["label_corr"] = [classify(a, b, c) for a, b, c in zip(ab.f1, ab.f2, ab.ft)]
ab_tab = ab[["config", "E_total", "f1", "f2", "ft", "A20", "reconstruction_rmse", "label_corr"]].copy()
ab_tab.to_csv(os.path.join(OUT, "corrected_ablation_table.csv"), index=False)
print("\n=== LOSS ABLATION (corrected, with f1) ===")
print(ab_tab.to_string(index=False))

# ---------------------------------------------------------------- seed study (10)
mgm = pd.read_csv(MAINGRID_MODE)
seedsel = mgm[(mgm.Re == 500) & (np.isclose(mgm.E_star, 0.25)) & (mgm.seed < 10)].copy()
seedsel["f1"] = seedsel["Efrac_0_0"]
seedsel["f2"] = seedsel["Efrac_1_0"]
# temporal = sum Efrac_*_j for j>=1
temp_cols = [c for c in seedsel.columns if c.startswith("Efrac_") and c.split("_")[-1] != "0"]
seedsel["ft"] = seedsel[temp_cols].sum(axis=1)
seedsel["label_corr"] = [classify(a, b, c) for a, b, c in zip(seedsel.f1, seedsel.f2, seedsel.ft)]
SEED_LOSS = {0: 1.377, 1: 1.377, 2: 1.374, 3: 1.372, 4: 1.394,
             5: 1.373, 6: 1.376, 7: 1.374, 8: 1.372, 9: 1.371}
seed_tab = seedsel[["seed", "E_total", "f1", "f2", "ft", "A20", "label_corr"]].sort_values("seed")
seed_tab["L_tot"] = seed_tab["seed"].map(SEED_LOSS)
seed_tab.to_csv(os.path.join(OUT, "corrected_seed_study.csv"), index=False)
print("\n=== SEED STUDY (Re=500,E*=0.25, seeds 0-9) ===")
print(seed_tab.to_string(index=False))
print("seed counts:", seed_tab.label_corr.value_counts().to_dict())

# ---------------------------------------------------------------- energy sweep
est = np.array([0.01, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00])
er = np.array([0.0540, 0.0790, 0.1183, 0.2534, 0.5006, 0.9756, 1.9333])
f2 = np.array([92.8, 92.2, 92.1, 92.5, 85.4, 86.3, 74.0])
ft = np.array([0.4, 0.5, 0.7, 1.2, 6.6, 9.2, 21.8])
f1 = 100.0 - f2 - ft  # remainder dominated by fundamental/other stationary modes
es_tab = pd.DataFrame({"E_star": est, "E_r": er, "f1_pct": f1, "f2_pct": f2, "ft_pct": ft})
es_tab.to_csv(os.path.join(OUT, "corrected_energy_sweep.csv"), index=False)
print("\n=== ENERGY SWEEP (Re=500) ===")
print(es_tab.to_string(index=False))

# ---------------------------------------------------------------- global summary
rows = []
for src, sub in g.groupby("source"):
    s = summarize(sub)
    if src == "C2":
        # C2 is Paper 3 reuse (registry only, not a Paper 2 result). The complete
        # 49-run campaign list is kept in runs_campaign2.csv; the per-run audit
        # global_reclassification.csv currently enumerates 47/49 C2 runs (the
        # lambda_var = 10 seeds 5-6 are missing there). Summary from the complete
        # list so the registry count (49) is preserved. f1 = 1 - f2 - ft.
        cr = pd.read_csv(C2_RUNS)
        cr["f1"] = 1.0 - cr.f_2 - cr.f_t
        cr["corr_label"] = [classify(a, b, c) for a, b, c in zip(cr.f1, cr.f_2, cr.f_t)]
        s = summarize(cr)
    rows.append({"source": src, **s})
glob = pd.DataFrame(rows)
glob.to_csv(os.path.join(OUT, "corrected_global_summary.csv"), index=False)
print("\n=== GLOBAL SUMMARY (all canonical runs) ===")
print(glob.to_string(index=False))

# reuse / duplication note (C1/C2 vs Paper 2)
print("\n=== C1 / C2 (Paper 3 reuse) ===")
for src in ["C1", "C2"]:
    sub = g[g.source == src]
    s = summarize(sub)
    if src == "C2":
        cr = pd.read_csv(C2_RUNS)
        cr["f1"] = 1.0 - cr.f_2 - cr.f_t
        cr["corr_label"] = [classify(a, b, c) for a, b, c in zip(cr.f1, cr.f_2, cr.f_t)]
        s = summarize(cr)
    print(src, s)
