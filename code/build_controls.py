# -*- coding: utf-8 -*-
"""
build_controls.py — Construit tous les contrôles du case study Re=500, E*=0.25.

Deux familles (séparation stricte PINN -> LBM, torch pour PINN) :

  Expérience A (contrôles réels du PINN, validation LBM) :
    uniform (lid plat, amplitude 1)
    sin_2pi (A2=0.68355, amplitude du contrôle PINN final campagne 13 seed 42)
    pinn_mean (moyenne temporelle des 15 seeds Paper2 (500,0.25))

  Case study mécanistique (Énergie EXACTE E=0.25 pour tous) :
    uniform_E025 :  U(x)=0.5                       ⟨U²⟩=0.25
    B1_E025      :  U(x)=A2·sin(2πx), A2=√(2E)=√0.5≈0.70710678
    B2_seed5_E025:  série U(x,t) complète du seed 5 (pure B2, f_t=99.8%),
                    renormalisée α=√(0.25/⟨U²⟩_x,t)
    B3_seed0_E025:  série U(x,t) complète du seed 0 (B3, f_2=52%, f_t=45%),
                    renormalisée α=√(0.25/⟨U²⟩_x,t)

Le paramètre E est défini par ⟨U_lid²⟩ (moyenne spatiale pour stationnaire,
spatio-temporelle pour B2/B3), sur la grille du LBM (trapèzes pondérés).

Sorties :
  data/lid_series/<name>_re500_e0.25.npz   (U_series (nt,N) ou U_mean (N,) + E mesuré)
  results/seed_controls_re500_e0.25.csv    (projection modale A_k0 + E par seed)
  results/case_controls_energy.csv         (E budgétaire vs E mesuré, α, A2, f_2/f_t)
  logs/build_controls.log
"""
import argparse, csv, json, os, shutil, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as fh:
    CFG = json.load(fh)
P1R = os.path.join(ROOT, CFG["paths"]["paper1_repo"])

import numpy as np
sys.path.insert(0, P1R)
import torch
import PINN_Lid_driven_reviewers as P

N_LBM = 256
NT_PER = 256
T_PERIOD = 2.0
E_BUDGET = CFG["case"]["energy_budget"]
B2_SEED = CFG["case"]["b2_seed"]
B3_SEED = CFG["case"]["b3_seed"]


def series_from_model(model, Re, nx=N_LBM, nt=NT_PER, t_period=T_PERIOD):
    xs = torch.linspace(0, P.LX, nx, dtype=torch.float64).view(-1, 1)
    ts = torch.linspace(0, t_period, nt, dtype=torch.float64).view(-1, 1)
    X, T = torch.meshgrid(xs[:, 0], ts[:, 0], indexing="ij")
    Xf, Tf = X.reshape(-1, 1), T.reshape(-1, 1)
    Re_t = torch.full_like(Xf, float(Re))
    with torch.no_grad():
        U = model.U_lid(Xf, Tf, Re_t).cpu().numpy().reshape(nx, nt)
    return U.T                                  # (nt, nx), lignes = t


def energy_of(series):
    """⟨U²⟩ pondéré (trapèzes) sur [0,T_PERIOD]×[0,1] (spatial pur si nt=1)."""
    nt, nx = series.shape
    dx = 1.0 / (nx - 1)
    wx = np.ones(nx); wx[0] = wx[-1] = 0.5
    if nt == 1:
        return float(np.sum(wx * series[0] ** 2) * dx)
    dt = T_PERIOD / (nt - 1)
    wt = np.ones(nt); wt[0] = wt[-1] = 0.5
    W = np.outer(wt, wx)
    return float(np.sum(W * series ** 2) * dx * dt / (1.0 * T_PERIOD))


def modal_coeffs(u, nx):
    xs = np.linspace(0, 1, nx)
    dx = 1.0 / (nx - 1); w = np.ones(nx); w[0] = w[-1] = 0.5
    return {f"A{k}0": 2.0 * np.sum(w * u * np.sin(k * np.pi * xs)) * dx for k in range(1, 9)}


def write_npz(outdir, name, series, E_meas):
    """series (nt, N) ; on écrit U_series + U_mean; npz unique par contrôle."""
    np.savez(os.path.join(outdir, f"{name}_re500_e0.25.npz"),
             U_series=series, U_mean=series.mean(axis=0),
             E_mean=float(np.mean(series ** 2)), E_measured=E_meas,
             N=N_LBM, NT_PER=series.shape[0], T_PERIOD=T_PERIOD,
             max_abs=float(np.abs(series).max()))


def main():
    t0 = time.time()
    log = open(os.path.join(ROOT, "logs", "build_controls.log"), "w", encoding="utf-8")
    def logp(m):
        print(m); log.write(m + "\n")

    Re = CFG["study"]["operation_point"]["Re"]
    out = os.path.join(ROOT, CFG["outputs"]["lid_series_dir"])
    res = os.path.join(ROOT, CFG["outputs"]["results_dir"])
    os.makedirs(out, exist_ok=True); os.makedirs(res, exist_ok=True)
    xs = np.linspace(0, 1, N_LBM)
    logp(f"[build_controls] Re={Re} E_budget={E_BUDGET} B2=seed{B2_SEED} B3=seed{B3_SEED}")

    # ============================================================ Expérience A
    np.savez(os.path.join(out, "uniform_re500_e0.25.npz"),
             U_series=np.ones((1, N_LBM)), U_mean=np.ones(N_LBM),
             E_mean=1.0, E_measured=1.0, N=N_LBM, NT_PER=1, T_PERIOD=1.0)
    logp("  [A] uniform OK  E=1.0")
    A2 = CFG["controls"]["sin_2pi"]["A2"]
    np.savez(os.path.join(out, "sin_2pi_re500_e0.25.npz"),
             U_series=(A2 * np.sin(2 * np.pi * xs))[None, :],
             U_mean=A2 * np.sin(2 * np.pi * xs), E_mean=A2 ** 2 / 2,
             E_measured=A2 ** 2 / 2, A2=A2, N=N_LBM, NT_PER=1, T_PERIOD=1.0)
    logp(f"  [A] sin_2pi OK  E={A2**2/2:.4f}")

    # ------------------------------------------------------- 15 seeds du PINN
    seed_dir = os.path.join(ROOT, CFG["paths"]["seed_models"])
    seeds = sorted([d for d in os.listdir(seed_dir) if d.startswith("seed_")],
                   key=lambda s: int(s.replace("seed_", "")))
    series_by_seed, means, rows = {}, [], []
    for s in seeds:
        mp = os.path.join(seed_dir, s, "model.pt")
        model = P.UltraPINN(arch_config=None, n_mx=P.N_MX, n_mt=P.N_MT,
                            lx=P.LX, ly=P.LY, basis_type="fourier").to("cpu")
        model.load_state_dict(torch.load(mp, map_location="cpu", weights_only=False))
        model.eval()
        ser = series_from_model(model, Re)
        series_by_seed[int(s.replace("seed_", ""))] = ser
        um = ser.mean(axis=0)
        means.append(um)
        A = modal_coeffs(um, N_LBM)
        E_m = energy_of(ser)
        rows.append({"seed": int(s.replace("seed_", "")),
                     "A2": round(A["A20"], 6), "A10": round(A["A10"], 6),
                     "A30": round(A["A30"], 6), "E_x_t": round(E_m, 6),
                     "E_mean_temporal": round(float(np.mean(um ** 2)), 6)})
    logp(f"  15 seeds : {','.join(str(r['seed']) for r in rows)}")

    # ------------------------------------------------ PINN(mean), Expérience A
    um_mean = np.mean(np.array(means), axis=0)
    np.savez(os.path.join(out, "pinn_mean_re500_e0.25.npz"),
             U_series=um_mean[None, :], U_mean=um_mean,
             E_mean=float(np.mean(um_mean ** 2)), E_measured=energy_of(um_mean[None, :]),
             n_seeds=len(seeds), N=N_LBM, NT_PER=1, T_PERIOD=1.0, raw_mean=um_mean)
    logp(f"  [A] pinn_mean OK  E={energy_of(um_mean[None, :]):.4f}")

    # ============================================== Case study (E=E_BUDGET)
    case_rows = [
        {"control": "uniform_E025", "type": "stationary",
         "series": np.full((1, N_LBM), np.sqrt(E_BUDGET)),
         "note": f"U=sqrt(E)={np.sqrt(E_BUDGET):.6f}"},
        {"control": "B1_E025", "type": "stationary",
         "series": (np.sqrt(2 * E_BUDGET) * np.sin(2 * np.pi * xs))[None, :],
         "note": f"A2=sqrt(2E)={np.sqrt(2*E_BUDGET):.8f}"},
        {"control": f"B3_seed{B3_SEED}_E025", "type": "temporal",
         "series": series_by_seed[B3_SEED],
         "note": f"seed {B3_SEED} complet (B3, f_2/f_t du produit PINN), α=sqrt(E/⟨U²⟩_x,t)"},
        {"control": f"B2_seed{B2_SEED}_E025", "type": "temporal",
         "series": series_by_seed[B2_SEED],
         "note": f"seed {B2_SEED} complet (B2 pur, f_t=0.998), α=sqrt(E/⟨U²⟩_x,t)"},
    ]
    energy_rows = []
    for cr in case_rows:
        ser = np.ascontiguousarray(cr["series"], dtype=np.float64)
        E0 = energy_of(ser)
        if cr["type"] == "temporal":
            alpha = float(np.sqrt(E_BUDGET / E0))
            ser = ser * alpha
            cr["alpha"] = alpha
        E_meas = energy_of(ser)
        write_npz(out, cr["control"], ser, E_meas)
        logp(f"  [case] {cr['control']:18s} E0={E0:.6f} α={cr.get('alpha','—')} "
             f"E_meas={E_meas:.8f} peak={np.abs(ser).max():.4f}")
        energy_rows.append({"control": cr["control"], "type": cr["type"],
                            "alpha": cr.get("alpha", 1.0), "E_budget": E_BUDGET,
                            "E_measured": E_meas, "E_0": E0,
                            "peak": float(np.abs(ser).max()), "note": cr["note"]})
    csvp = os.path.join(res, "case_controls_energy.csv")
    with open(csvp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["control", "type", "E_budget", "E_measured",
                                           "E_0", "alpha", "peak", "note"])
        w.writeheader(); w.writerows(energy_rows)
    logp(f"  case_controls_energy.csv écrit")

    # ------------------------------------------------------ seeds CSV + copy
    with open(os.path.join(res, "seed_controls_re500_e0.25.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["seed", "A2", "A10", "A30", "E_x_t",
                                           "E_mean_temporal"])
        w.writeheader()
        w.writerows(rows)
        Am = modal_coeffs(um_mean, N_LBM)
        w.writerow({"seed": "mean", "A2": round(Am["A20"], 6),
                    "A10": round(Am["A10"], 6), "A30": round(Am["A30"], 6),
                    "E_x_t": round(energy_of(um_mean[None, :]), 6),
                    "E_mean_temporal": round(float(np.mean(um_mean ** 2)), 6)})
    src = os.path.join(ROOT, CFG["paths"]["fourier_series_paper1"])
    dst_in = os.path.join(ROOT, "data", "inputs", os.path.basename(src))
    if os.path.isfile(src) and os.path.abspath(src) != os.path.abspath(dst_in):
        shutil.copy2(src, dst_in)
        logp("  fourier_Re500.npz copié dans data/inputs")
    elif os.path.isfile(src):
        logp("  fourier_Re500.npz déjà présent dans data/inputs (copie inutile)")
    logp(f"[done] en {time.time()-t0:.0f}s")
    log.close()


if __name__ == "__main__":
    main()