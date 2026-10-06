# -*- coding: utf-8 -*-
"""
run_case_study.py — Solve LBM-MRT pour chacun des trois contrôles au point
Re=500, E*=0.25, puis calcule les diagnostics physiques et écrit :
  results/scalars_<control>.json, results/fields_<control>.npz,
  results/case_study_summary.csv   (table EPS_xx/EPS_yy/EPS_xy/EPS/K/Z/P_lid)
  logs/conv_<control>.csv  (historique itération/erreur/temps)

Usage :  py -3.11 run_case_study.py [--N 128] [--max-iter 1200000] [--tol 1e-8]
                   [--controls uniform sin_2pi pinn_mean] [--resume]

Reproductibilité : exécutez `py -3.11 run_all.py` après avoir installé les
dépendances (numba, numpy, torch pour build_controls) — voir README.md.
"""
import argparse, csv, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as fh:
    CFG = json.load(fh)

import numpy as np

sys.path.insert(0, os.path.join(ROOT, CFG["paths"]["paper1_repo"], "lbm_mrt_validation"))
import lbm_mrt_pinn_validation as LBM

import diagnostics as D

RESULTS = os.path.join(ROOT, CFG["outputs"]["results_dir"])
LID = os.path.join(ROOT, CFG["outputs"]["lid_series_dir"])
LOGS = os.path.join(ROOT, CFG["outputs"]["logs_dir"])
for d in (RESULTS, LID, LOGS):
    os.makedirs(d, exist_ok=True)


def load_series(control, N):
    """Charge le NPZ data/lid_series/<control>_re500_e0.25.npz et ré-échantillonne
    (period, N) ; period=1 pour un contrôle stationnaire."""
    p = os.path.join(LID, f"{control}_re500_e0.25.npz")
    with np.load(p) as z:
        if "U_series" in z:
            ser = z["U_series"]                      # (nt, nx) ou (1, nx)
        else:
            ser = z["U_mean"][None, :]
    ser = np.asarray(ser, dtype=np.float64)
    nt, nx0 = ser.shape
    x0 = np.linspace(0, 1, nx0); x = np.linspace(0, 1, N)
    ser_r = np.array([np.interp(x, x0, ser[t]) for t in range(nt)])   # (nt, N)
    if nt == 1:
        ser_r = ser_r[0]                             # période 1 : (N,)
    return ser_r


def set_lid(solver, series):
    """Remplace le lid du solver par une série (nt, N) ou (N,)."""
    series = np.ascontiguousarray(series, dtype=np.float64)
    if series.ndim == 1:
        period = 1
        ud = series[None, :]
    else:
        period = series.shape[0]
        ud = series
    solver.period = period
    solver.U_lid_norm = ud
    solver.U_lid2d = solver.U_ref * ud
    return solver


def run_one(control, N, max_iter, tol, verbose=True):
    tag = f"{control}_Re{CFG['study']['operation_point']['Re']}_N{N}"
    solver = LBM.LBM_MRT_Solver(CFG["study"]["operation_point"]["Re"], N,
                                lid_profile="uniform", max_iter=max_iter, tol=tol,
                                stat_cycles=CFG["lbm"]["stat_cycles"])
    ser = load_series(control, N)
    set_lid(solver, ser)

    convfile = os.path.join(LOGS, f"conv_{control}.csv")
    t0 = time.time()
    solver.run(verbose=verbose, logfile=convfile)
    dt = time.time() - t0
    print(f"[run_one] {control} N={N} iters={solver.iters} resid={solver.last_residual:.2e} "
          f"period={solver.period} time={dt:.0f}s")
    solver.compute_integrals()                       # fixe solver.K_fluct -> R_K

    D.compute(solver, tag, RESULTS)
    return solver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=CFG["lbm"]["N_default"])
    ap.add_argument("--max-iter", type=int, default=CFG["lbm"]["max_iter_default"])
    ap.add_argument("--tol", type=float, default=CFG["lbm"]["tol_default"])
    ap.add_argument("--controls", nargs="+",
                    default=CFG["study"]["controls"])
    args = ap.parse_args()

    summary = []
    for c in args.controls:
        s = run_one(c, args.N, args.max_iter, args.tol)
        tag = f"{c}_Re500_N{args.N}"
        with open(os.path.join(RESULTS, f"scalars_{tag}.json"), encoding="utf-8") as fh:
            sc = json.load(fh)
        summary.append(sc)

    # Ratio par rapport au cas de référence uniform_E025 (même budget d'énergie)
    eps_u = None
    for s in summary:
        if "uniform_E025" in s["control"]:
            eps_u = s["eps"]; break
    if eps_u is None:
        up = os.path.join(RESULTS, "scalars_uniform_E025_Re500_N%d.json" % args.N)
        if os.path.exists(up):
            eps_u = json.load(open(up, encoding="utf-8"))["eps"]
    csvp = os.path.join(RESULTS, "case_study_summary.csv")
    if eps_u is not None:
        for s in summary:
            s["eps_over_eps_uniform"] = s["eps"] / eps_u
    with open(csvp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["control", "Re", "N", "period", "stat_iters",
                                           "iters",
                                           "last_residual",
                                           "eps", "eps_xx", "eps_yy", "eps_xy",
                                           "frac_xx", "frac_yy", "frac_xy",
                                           "eps_over_eps_uniform",
                                           "K", "Z", "P_lid", "K_fluct",
                                           "P_lid_balance", "R_K"])
        w.writeheader()
        for s in summary:
            row = {}
            for k, v in s.items():
                if k not in w.fieldnames:
                    continue
                if v is None:
                    row[k] = ""
                elif isinstance(v, float):
                    row[k] = f"{v:.6e}"
                else:
                    row[k] = v
            w.writerow(row)
    print(f"[done] summary -> {csvp}")


if __name__ == "__main__":
    main()