# -*- coding: utf-8 -*-
"""
Campagne 4 (brenches.txt §VII / matrice : « B2 temporal averaging »).
Contrôle B2 déjà sélectionné (B2_seed5_E025, period=256, T=2.0), run LBM long
SANS break de tolérance (tol=0) pour que le solveur accumule ses séries
par-cycle (cyc_eps/cyc_K/cyc_Kf/cyc_Rk + ring cyc_tail des champs moyens).

Reconstruction des fenêtres T et 2T (cycles) sur la partie post-settle :
  fenêtre T  = derniers n_T cycles      (n_T = cycles_2T // 2)
  fenêtre 2T = derniers 2*n_T cycles
  pour chaque fenêtre : <K>, sigma_K, <e>, sigma_e, R_K = <Kf>/<K> (+ mean Rk)
Critère de clôture (brenches) :
  R_K(T) ~= R_K(2T)  =>  instationnarité statistiquement persistante
  sinon               =>  régime fortement transitoire sur la fenêtre étudiée.
Dans les DEUX cas le manuscrit ne compare pas <e_B2> à B1 comme si B2 était
stationnaire (convention conservée dans le texte de discussion).

Usage :
  py fenetres_B2.py --N 256 --max-iter 640000       (tol force a 0)
  py fenetres_B2.py --N 512 --max-iter 1280000
Sorties :
  results/fenetres_B2_Re500_N<N>.json   (fenetres T/2T + scalaires)
  results/fenetres_B2_Re500_N<N>.npz    (series par-cycle + champ moyen final)
  logs/sweep_energy_matched_... non touche ; log dedie logs/fenetres_B2_<N>.log
"""
import argparse, json, os, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as fh:
    CFG = json.load(fh)

import numpy as np

import os, sys
sys_path = os.path.join(ROOT, CFG["paths"]["paper1_repo"], "lbm_mrt_validation")
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)
import lbm_mrt_pinn_validation as LBM
import run_case_study as R          # load_series, set_lid, RESULTS/LOGS paths

RESULTS = os.path.join(ROOT, CFG["outputs"]["results_dir"])
LOGS = os.path.join(ROOT, CFG["outputs"]["logs_dir"])
for d in (RESULTS, LOGS):
    os.makedirs(d, exist_ok=True)

CONTROL = "B2_seed5_E025"
RE = CFG["study"]["operation_point"]["Re"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--max-iter", type=int, default=640000,
                    help="iters du run long (tol=0 -> jamais de break)")
    ap.add_argument("--cycles-2T", type=int, default=128,
                    help="2T = dernier 2*cycles_2T cycles ; T = cycles_2T")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    N, mi = args.N, args.max_iter
    n_2T = args.cycles_2T
    n_T = 2 * n_2T

    tag = f"{CONTROL}_Re{RE}_N{N}"
    logp = open(os.path.join(LOGS, f"fenetres_B2_{N}.log"), "w", encoding="utf-8")
    def log(m):
        print(m); logp.write(m + "\n"); logp.flush()

    log(f"[B2 windows] control={CONTROL} Re={RE} N={N} max_iter={mi} "
        f"tol=0(nobreak) T={n_2T}cyc 2T={n_T}cyc")

    solver = LBM.LBM_MRT_Solver(RE, N, lid_profile="uniform",
                                max_iter=mi, tol=0.0,
                                stat_cycles=CFG["lbm"]["stat_cycles"])
    ser = R.load_series(CONTROL, N)
    R.set_lid(solver, ser)
    t0 = time.time()
    solver.run(verbose=args.verbose, logfile=os.path.join(LOGS, f"conv_{CONTROL}_win.csv"))
    dt = time.time() - t0
    log(f"[run] period={solver.period} iters={solver.iters} "
        f"mean_count={solver.mean_count} time={dt:.0f}s")

    # ---- series par-cycle accumulees par le solveur (apres settle_at) ----
    cyc_eps = np.asarray(solver.cyc_eps)
    cyc_K = np.asarray(solver.cyc_K)
    cyc_Kf = np.asarray(solver.cyc_Kf)
    cyc_Rk = np.asarray(solver.cyc_Rk)
    n_cyc = len(cyc_eps)
    log(f"[cycles] n_cyc_total={n_cyc}  (seuil fenetres: 2T={n_T}, T={n_2T})")
    if n_cyc <= n_T:
        log("[WARNING] trop peu de cycles pour 2T -> reduction auto a n_cyc")
        n_T = n_cyc
        n_2T = n_cyc // 2
        log(f"[WARNING] 2T={n_T}, T={n_2T} (ajuste)")
    if n_T < n_2T * 2:
        n_T = n_2T * 2

    def stats_window(a):
        return dict(mean=float(np.mean(a)), std=float(np.std(a)),
                    min=float(np.min(a)), max=float(np.max(a)))

    w2T = slice(-n_T, None)
    wT = slice(-n_2T, None)
    res = {
        "control": CONTROL, "Re": RE, "N": N, "period": int(solver.period),
        "iters": int(solver.iters), "n_cycles_total": int(n_cyc),
        "T_cycles": int(n_2T), "T2_cycles": int(n_T),
        "window_T": {
            "K": stats_window(cyc_K[wT]),
            "K_fluct": stats_window(cyc_Kf[wT]),
            "eps": stats_window(cyc_eps[wT]),
            "R_K_ratio_of_means": float(np.mean(cyc_Kf[wT]) / np.mean(cyc_K[wT])),
            "R_K_mean_of_ratios": float(np.mean(cyc_Rk[wT])),
            "R_K_std": float(np.std(cyc_Rk[wT])),
        },
        "window_2T": {
            "K": stats_window(cyc_K[w2T]),
            "K_fluct": stats_window(cyc_Kf[w2T]),
            "eps": stats_window(cyc_eps[w2T]),
            "R_K_ratio_of_means": float(np.mean(cyc_Kf[w2T]) / np.mean(cyc_K[w2T])),
            "R_K_mean_of_ratios": float(np.mean(cyc_Rk[w2T])),
            "R_K_std": float(np.std(cyc_Rk[w2T])),
        },
        "R_K_persistence": abs(np.mean(cyc_Rk[wT]) - np.mean(cyc_Rk[w2T]))
                           / max(np.mean(cyc_Rk[w2T]), 1e-30),
        "window_time_s": dt,
    }
    # classification brenches : persistant si |R_K(T)-R_K(2T)|/R_K(2T) < 15%
    res["verdict"] = ("PERSISTANT" if res["R_K_persistence"] < 0.15
                      else "TRANSIENT_ON_WINDOW")

    with open(os.path.join(RESULTS, f"fenetres_B2_Re{RE}_N{N}.json"),
              "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    np.savez_compressed(os.path.join(RESULTS, f"fenetres_B2_Re{RE}_N{N}.npz"),
                        cyc_eps=cyc_eps, cyc_K=cyc_K, cyc_Kf=cyc_Kf, cyc_Rk=cyc_Rk,
                        period=solver.period, N=N, Re=RE,
                        U_lid_imposed_first=np.asarray(
                            solver.U_lid_norm.reshape(solver.period, N)[0]))

    log("=" * 72)
    log(f"FENETRE T   ({n_2T} cyc):  <K>={res['window_T']['K']['mean']:.4e} "
        f"sigma_K={res['window_T']['K']['std']:.4e}  "
        f"<eps>={res['window_T']['eps']['mean']:.4e} "
        f"sigma_e={res['window_T']['eps']['std']:.4e}  "
        f"R_K={res['window_T']['R_K_mean_of_ratios']:.4f}")
    log(f"FENETRE 2T  ({n_T} cyc):  <K>={res['window_2T']['K']['mean']:.4e} "
        f"sigma_K={res['window_2T']['K']['std']:.4e}  "
        f"<eps>={res['window_2T']['eps']['mean']:.4e} "
        f"sigma_e={res['window_2T']['eps']['std']:.4e}  "
        f"R_K={res['window_2T']['R_K_mean_of_ratios']:.4f}")
    log(f"persistance |dR_K|/R_K(2T) = {res['R_K_persistence']:.3%}  "
        f"-> {res['verdict']}")
    log(f"-> JSON {os.path.join(RESULTS, f'fenetres_B2_Re{RE}_N{N}.json')}")
    logp.close()


if __name__ == "__main__":
    main()