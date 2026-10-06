# -*- coding: utf-8 -*-
"""
sweep_energy_matched.py — Comparaison ENERGY-MATCHED
    B1  :  U(x) = A2·sin(2πx)             (E* = A2²/2)
    Ref :  U(x) = U0 = A2/√2 = √E*        (même ⟨U²⟩ = E*)
Re=500, AR=1. Répond : à énergie d'actuation égale, la 2e harmonique
stationnaire est-elle dissipativement favorable, neutre ou défavorable ?

Logique de reprise :
  - runs B1sweep_* déjà effectués (N=128/256) réutilisés par valeur de A2 ;
  - scalars uniform_E025 réutilisés pour E*=0.25 (U0=0.5) s'ils existent ;
  - tout run manquant est exécuté puis archivé (scalars_*.json) — relancer le
    script ne relance que ce qui manque.

Usage :
  py -3.11 sweep_energy_matched.py --N 256 --tol 1e-9 --max-iter 5000000 \
       --a2 0.60 0.63024 0.64855 0.650 0.658 0.666 0.68355 0.70711
  py -3.11 sweep_energy_matched.py --N 512 --tol 1e-9 --max-iter 2000000 --a2 0.658

Sorties par grille N :
  results/energy_matched_<N>.csv        (A2, E*, eps_B1, eps_uniform, ratio, tokens)
  figures/fig_energy_matched.pdf/.png   (ratio vs E*, ligne y=1, croisement)
  logs/sweep_energy_matched_<N>.log
"""
import argparse, csv, json, os, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
import numpy as np
import run_case_study as R

FIG_DIR = os.path.join(ROOT, R.CFG["outputs"]["figures_dir"])
LOG_DIR = os.path.join(ROOT, R.CFG["outputs"]["logs_dir"])
RES_DIR = os.path.join(ROOT, R.CFG["outputs"]["results_dir"])
Re = R.CFG["study"]["operation_point"]["Re"]

DEFAULT_A2 = [0.60, 0.63024, 0.64855, 0.650, 0.658, 0.666, 0.68355, 0.70710678]


def tag_of(x):
    s = f"{x:.6f}".rstrip("0")
    if s.endswith("."):
        s += "0"
    return s.replace(".", "p")


def scalars_path(control, N):
    return os.path.join(RES_DIR, f"scalars_{control}_Re{Re}_N{N}.json")


def get_sc(control, N):
    with open(scalars_path(control, N), encoding="utf-8") as fh:
        return json.load(fh)


def write_series(fname, u):
    xs = np.linspace(0, 1, len(u))
    out = os.path.join(ROOT, R.CFG["outputs"]["lid_series_dir"])
    np.savez(os.path.join(out, f"{fname}_re500_e0.25.npz"),
             U_series=np.asarray(u)[None, :], U_mean=np.asarray(u),
             E_mean=float(np.mean(np.asarray(u) ** 2)),
             E_measured=float(np.mean(np.asarray(u) ** 2)),
             N=len(u), NT_PER=1, T_PERIOD=1.0,
             max_abs=float(np.abs(np.asarray(u)).max()))


def b1_series(a2, N, k=2):
    xs = np.linspace(0, 1, N)
    return (a2 * np.sin(k * np.pi * xs)).astype(np.float64)


def uniform_series(U0, N):
    return np.full(N, U0, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--tol", type=float, default=R.CFG["lbm"]["tol_production"])
    ap.add_argument("--max-iter", type=int, default=R.CFG["lbm"]["max_iter_production"])
    ap.add_argument("--a2", nargs="+", type=float, default=DEFAULT_A2)
    ap.add_argument("--k", type=int, default=2,
                    help="spatial harmonic of the 'B-side' sin(k*pi*x) (default 2 = B1)")
    ap.add_argument("--verbose", action="store_true",
                    help="print solver progress (it= / DeltaU=) during runs")
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()
    N, tol, mi, k = args.N, args.tol, args.max_iter, args.k

    logp = open(os.path.join(LOG_DIR, f"sweep_energy_matched_{N}_k{k}.log"), "w",
                encoding="utf-8")
    def log(m):
        print(m); logp.write(m + "\n"); logp.flush()

    log(f"[EM] Re={Re} N={N} tol={tol} max_iter={mi} k={k} nA2={len(args.a2)}")

    rows = []
    for a2 in args.a2:
        E = a2 * a2 / 2.0
        U0 = a2 / np.sqrt(2.0)
        tagb, tagu = tag_of(a2), tag_of(U0)
        tokens = []

        # ---- B-side : réutilise B1sweep_<tag>/B1em_<tag> si k=2, sinon S{k}em_<tag> ----
        cb = f"B1sweep_{tagb}" if k == 2 else None
        if cb is None or not os.path.exists(scalars_path(cb, N)):
            cb = (f"B1em_{tagb}" if k == 2 else f"S{k}em_{tagb}")
        if not os.path.exists(scalars_path(cb, N)):
            write_series(cb, b1_series(a2, N, k))
            t0 = time.time()
            R.run_one(cb, N, mi, tol, verbose=args.verbose)
            dt = time.time() - t0
            tokens.append(f"B={dt:.0f}s")
        else:
            tokens.append("B=reused")
        sc_b = get_sc(cb, N)
        eps_b = sc_b["eps"]

        # ---- uniform energy-matched : uniform_E025 si U0=0.5 sinon UNem_<tag> ----
        cu = "uniform_E025"
        if not (abs(U0 - 0.5) < 1e-9 and os.path.exists(scalars_path(cu, N))):
            cu = f"UNem_{tagu}"
        if not os.path.exists(scalars_path(cu, N)):
            write_series(cu, uniform_series(U0, N))
            t0 = time.time()
            R.run_one(cu, N, mi, tol, verbose=args.verbose)
            dt = time.time() - t0
            tokens.append(f"U={dt:.0f}s")
        else:
            tokens.append("U=reused")
        sc_u = get_sc(cu, N)
        eps_u = sc_u["eps"]

        ratio = eps_b / eps_u
        rows.append({"A2": a2, "E_star": E, "U0": U0, "control_B1": cb,
                     "control_UN": cu, "eps_B1": eps_b, "eps_uniform": eps_u,
                     "ratio": ratio, "tokens": "+".join(tokens)})
        log(f"  A2={a2:.5f} E*={E:.5f} B1={cb}({eps_b:.4e}) "
            f"UN={cu}({eps_u:.4e}) ratio={ratio:.4f} [{'+'.join(tokens)}]")
    logp.close()

    # ---- CSV (version finale, tous points) ----
    a2s = np.array([r["A2"] for r in rows]); es = np.array([r["E_star"] for r in rows])
    rs = np.array([r["ratio"] for r in rows])
    csvp = os.path.join(RES_DIR, f"energy_matched_{N}_k{k}.csv")
    order = np.argsort(a2s)
    with open(csvp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["A2", "E_star", "U0_uniform", "eps_B1", "eps_uniform",
                    "ratio_B1_over_uniform", "control_B1", "control_UN", "tokens"])
        for i in order:
            r = rows[i]
            w.writerow([f"{r['A2']:.6f}", f"{r['E_star']:.6f}", f"{r['U0']:.6f}",
                        f"{r['eps_B1']:.6e}", f"{r['eps_uniform']:.6e}",
                        f"{r['ratio']:.6f}", r["control_B1"], r["control_UN"],
                        r["tokens"]])
    log = open(os.path.join(LOG_DIR, f"sweep_energy_matched_{N}.log"), "a",
               encoding="utf-8")
    log.write(f"[csv] -> {csvp}\n"); log.close()

    # ---- croisement de ratio=1 (interpolation linéaire) ----
    cross = None
    if np.all(np.diff(rs) > 0) and rs.min() < 1.0 < rs.max():
        a2c = float(np.interp(1.0, rs, a2s))
        cross = {"A2_star": a2c, "E_star": a2c * a2c / 2.0}
    elif np.all(np.diff(rs) < 0) and rs.min() < 1.0 < rs.max():
        a2c = float(np.interp(1.0, rs[::-1], a2s[::-1]))
        cross = {"A2_star": a2c, "E_star": a2c * a2c / 2.0}
    with open(os.path.join(LOG_DIR, f"sweep_energy_matched_{N}.log"), "a",
              encoding="utf-8") as lg:
        lg.write(f"[crossing] {cross if cross else 'NO CROSSING in range'}\n")

    if args.no_fig:
        return rows, cross
    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # overlay de la série N=256 (fond gris) pour une figure de confirmation N=512
    ref_es, ref_rs = None, None
    if N == 512:
        ref_csv = os.path.join(RES_DIR, f"energy_matched_256_k{k}.csv")
        if os.path.exists(ref_csv):
            with open(ref_csv, newline="", encoding="utf-8") as fh:
                ref_rows = list(csv.DictReader(fh))
            ref_es = [float(r["E_star"]) for r in ref_rows]
            ref_rs = [float(r["ratio_B1_over_uniform"]) for r in ref_rows]

    fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    if ref_es is not None:
        ax.plot(ref_es, ref_rs, "s--", color="#9db4c7", ms=5, lw=1.2,
                label=rf"$\epsilon_{{A_2\sin({k}\pi x)}}/\epsilon_{{unif}}$ "
                      rf"(energy-matched) $N=256$")
    ax.plot(es, rs, "o-", color="#1f4e79", ms=6, lw=1.9,
            label=rf"$\epsilon_{{A_2\sin({k}\pi x)}}/\epsilon_{{unif}}$ "
                  rf"(energy-matched) $N={N}$")
    ax.axhline(1.0, color="#c00000", ls="--", lw=1.4,
               label=r"energy-matched neutral "
                     r"($\epsilon_{ctrl}=\epsilon_{unif}$)")
    ax.axvline(0.25, color="#38b764", ls=":", lw=1.4,
               label=r"operation point $E^*=0.25$")
    lower = min(rs.min(), 1.0, *(ref_rs if ref_rs else ()))
    upper = max(rs.max(), 1.0, *(ref_rs if ref_rs else ()))
    dy = upper - lower
    ax.set_xlim(min(es.min(), 0.25) - 0.010, max(es.max(), 0.25) + 0.015)
    ax.set_ylim(lower - 0.10 * dy, upper + 0.10 * dy)
    ax.set_xlabel(r"Actuation energy $E^* = \langle U^2_{lid}\rangle$")
    ax.set_ylabel(r"$\epsilon_{A_2\sin(k\pi x)}/\epsilon_{\mathrm{uniform}}$ (energy-matched)")
    ax.set_title(rf"$A_2\sin({k}\pi x)$ vs uniform lid, Re=500, N={N}")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8.5)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig_energy_matched_k{k}_N{N}.{ext}"), dpi=150)
    with open(os.path.join(LOG_DIR, f"sweep_energy_matched_{N}_k{k}.log"), "a",
              encoding="utf-8") as lg:
        lg.write(f"[done] figures/fig_energy_matched_k{k}_N{N}.pdf/png\n")
    return rows, cross


if __name__ == "__main__":
    main()
