# -*- coding: utf-8 -*-
"""
sweep_b1_a2.py — Sensibilité du contrôle stationnaire B1 = A2·sin(2πx) à son
amplitude A2 (Re=500, AR=1, même point de fonctionnement que le case study).

Pour chaque A2 (E = A²/2, NON renormalisé : l'amplitude est le paramètre libre),
on résout le LBM-MRT stationnaire, calcule les diagnostics et on collecte ε.
Objectif : cerner la valeur de croisement A2* telle que ε_B1(A2*) = ε_uniform(0.5),
i.e. la marge d'amplitude autour du point de fonctionnement E=0.25 (A2=√0.5).

Usage :
  py -3.11 sweep_b1_a2.py --N 128 --tol 1e-8 --max-iter 1200000   # scan rapide
  py -3.11 sweep_b1_a2.py --N 256 --tol 1e-9 --max-iter 5000000   # confirmation

Sorties :
  data/lid_series/B1sweep_a2_<v>_re500_e0.25.npz   (séries stationnaires)
  results/sweep_b1_A2_<N>.csv                      (ε, ε/ε_uniform, E, Z, frac_xy)
  figures/fig_sweep_b1.pdf + .png                   (ε(A2), croisement A2*, EA2=√0.5)
  logs/sweep_b1_a2_<N>.log
"""
import argparse, csv, json, os, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
import numpy as np
import run_case_study as R   # réutilise config, solver, load_series, run_one
import diagnostics as D

FIG_DIR = os.path.join(ROOT, R.CFG["outputs"]["figures_dir"])
LOG_DIR = os.path.join(ROOT, R.CFG["outputs"]["logs_dir"])
RES_DIR = os.path.join(ROOT, R.CFG["outputs"]["results_dir"])
Re = R.CFG["study"]["operation_point"]["Re"]
E_REF = R.CFG["case"]["energy_budget"]
A2_OP = float(np.sqrt(2.0 * E_REF))          # = √0.5, point de fonctionnement
N_LBM = 256

DEFAULT_A2 = [0.45, 0.55, 0.60, 0.64855, 0.68355, A2_OP, 0.75, 0.85, 1.00]


def tag_of(a2):
    s = f"{a2:.6f}".rstrip("0")
    if s.endswith("."):
        s += "0"
    return s.replace(".", "p")


def make_series(a2, N):
    xs = np.linspace(0, 1, N)
    u = (a2 * np.sin(2 * np.pi * xs)).astype(np.float64)
    out = os.path.join(ROOT, R.CFG["outputs"]["lid_series_dir"])
    np.savez(os.path.join(out, f"B1sweep_{tag_of(a2)}_re500_e0.25.npz"),
             U_series=u[None, :], U_mean=u,
             E_mean=float(np.mean(u ** 2)), E_measured=a2 * a2 / 2.0,
             A2=float(a2), N=N, NT_PER=1, T_PERIOD=1.0,
             max_abs=float(np.abs(u).max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=R.CFG["lbm"]["N_default"])
    ap.add_argument("--tol", type=float, default=R.CFG["lbm"]["tol_default"])
    ap.add_argument("--max-iter", type=int, default=R.CFG["lbm"]["max_iter_default"])
    ap.add_argument("--a2", nargs="+", type=float, default=DEFAULT_A2)
    args = ap.parse_args()
    N, tol, mi = args.N, args.tol, args.max_iter

    os.makedirs(FIG_DIR, exist_ok=True)
    logp = open(os.path.join(LOG_DIR, f"sweep_b1_a2_{N}.log"), "w", encoding="utf-8")
    def log(m):
        print(m); logp.write(m + "\n"); logp.flush()

    log(f"[sweep] Re={Re} N={N} tol={tol} npts={len(args.a2)} "
        f"A2_op={A2_OP:.6f} (E={E_REF})")

    # référence uniform_E025 (existant)
    up = os.path.join(RES_DIR, f"scalars_uniform_E025_Re{Re}_N{N}.json")
    if not os.path.exists(up):
        up = os.path.join(RES_DIR, f"scalars_uniform_E025_Re500_N{N}.json")
    eps_ref = json.load(open(up, encoding="utf-8"))["eps"]
    log(f"  eps_ref(uniform U=0.5) = {eps_ref:.6e}")

    # construction des séries + résolution LBM
    rows = []
    for a2 in args.a2:
        make_series(a2, N_LBM)
        control = f"B1sweep_{tag_of(a2)}"
        t0 = time.time()
        solver = R.run_one(control, N, mi, tol, verbose=False)
        dt = time.time() - t0
        tag = f"{control}_Re{Re}_N{N}"
        with open(os.path.join(RES_DIR, f"scalars_{tag}.json"), encoding="utf-8") as fh:
            sc = json.load(fh)
        rows.append({"A2": a2, "E": a2 * a2 / 2.0, "reps": None, "solver": None,
                     "eps": sc["eps"], "eps_xx": sc["eps_xx"], "eps_yy": sc["eps_yy"],
                     "eps_xy": sc["eps_xy"], "frac_xy": sc["frac_xy"],
                     "K": sc["K"], "Z": sc["Z"], "P_lid": sc["P_lid"],
                     "P_lid_balance": sc["P_lid_balance"],
                     "iters": solver.iters, "resid": solver.last_residual,
                     "time_s": round(dt, 1)})
        log(f"  A2={a2:.5f} E={a2*a2/2:.5f} eps={sc['eps']:.4e} "
            f"eps/ref={sc['eps']/eps_ref:.4f} iters={solver.iters} ({dt:.0f}s)")

    # croisement ε(A2) = ε_ref (interpolation linéaire, ε croissant en A2)
    a2s = np.array([r["A2"] for r in rows])
    es = np.array([r["eps"] for r in rows])
    iop = int(np.argmin(np.abs(a2s - A2_OP)))
    eps_op = es[iop]
    a2_star = None
    if np.all(np.diff(es) > 0) and es.min() < eps_ref < es.max():
        a2_star = float(np.interp(eps_ref, es, a2s))
    log(f"[crossing] A2* = {a2_star if a2_star is not None else 'HORS INTERVALLE'} "
        f"(E* = {((a2_star**2)/2 if a2_star is not None else float('nan')):.5f}) | "
        f"marge vs A2_op √0.5 : x{(A2_OP / a2_star) if a2_star else float('nan'):.3f}")

    for r in rows:
        r["eps_over_eps_uniform"] = r["eps"] / eps_ref
    csvp = os.path.join(RES_DIR, f"sweep_b1_A2_{N}.csv")
    with open(csvp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["A2", "E", "eps", "eps_over_eps_uniform",
                                           "eps_xx", "eps_yy", "eps_xy", "frac_xy",
                                           "K", "Z", "P_lid", "P_lid_balance",
                                           "iters", "resid", "time_s"])
        w.writeheader()
        for r in rows:
            row = {k: r[k] for k in w.fieldnames}
            for k in ("eps", "eps_over_eps_uniform", "eps_xx", "eps_yy", "eps_xy",
                      "frac_xy", "K", "Z", "P_lid", "P_lid_balance", "iters",
                      "resid", "time_s"):
                if isinstance(row[k], float):
                    row[k] = f"{row[k]:.6e}"
            w.writerow(row)
    log(f"[done] csv -> {csvp}")

    # --- figure : ε(A2) + ref uniform + croisement + point de fonctionnement
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    ax.plot(a2s, es, "o-", color="#1f4e79", ms=5, lw=1.6,
            label=r"$B_1$ : $\epsilon(A_2\sin 2\pi x)$")
    ax.axhline(eps_ref, color="#c00000", ls="--", lw=1.4,
               label=r"Uniform lid $U=0.5$ ($\epsilon_{ref}$)")
    ax.axvline(A2_OP, color="#38b764", ls=":", lw=1.4)
    ax.plot(A2_OP, eps_op, "o", color="#38b764", ms=6,
            label=r"$A_2=\sqrt{2E^*}=\sqrt{0.5}$ (operating point)")
    if a2_star is not None:
        ax.axvline(a2_star, color="#6f42c1", ls="-.", lw=1.4)
        ax.annotate(r"A$_{2}^{*}$ = " + f"{a2_star:.4f}\n" + r"(E$^{*}$ = "
                    + f"{a2_star**2/2:.4f})", xy=(a2_star, eps_ref),
                    xytext=(0.62, eps_ref * 1.30),
                    arrowprops=dict(arrowstyle="->", color="#6f42c1"), color="#6f42c1")
    ax.set_xlabel(r"Amplitude $A_2$ of $A_2\sin(2\pi x)$")
    ax.set_ylabel(r"Dissipation $\epsilon = \int_\Omega \Phi\, d\Omega$")
    ax.set_title(r"$B_1$ stationary control, Re=500, N=" + str(N))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8.5)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig_sweep_b1.{ext}"), dpi=150)
    log(f"[done] fig -> figures/fig_sweep_b1.pdf/png")
    logp.close()


if __name__ == "__main__":
    main()