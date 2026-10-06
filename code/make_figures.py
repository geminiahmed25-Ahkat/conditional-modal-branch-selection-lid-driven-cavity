# -*- coding: utf-8 -*-
"""
make_figures.py — Figures du case study mécanistique Re=500, E*=0.25
(matplotlib Agg, labels EN ANGLAIS, fidélité géométrique requise).

Exigences d'affichage (implémentées ici) :
  - cavité AR=1 : coordonnées EXACTES x,y ∈ [0,1] sur les panneaux de champ
    (aspect='equal', xlim/ylim imposés, ticks [0,0.5,1] sur tous les axes) ;
  - échelles identifiables : colorbars par panneau (vorticity, échelles propres)
    et colorbar partagée log10 pour Phi ; annotations max|u|, max|ω| ;
  - pas de superposition : constrained_layout, font sizes ≤10, légendes et
    colorbars placées hors-zone, pas de titre long sur les axes ;
  - la ligne lid (row 0) ne PARTAGE PAS l'axe y avec les lignes de champ.

Figures : fig_lid_controls, fig_composite_4x3, fig_quantitative, fig_tau_w
Lithographie fig_summary_numbers.csv. Usage : py -3.11 make_figures.py [--dpi 150]
"""
import argparse, csv, json, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as fh:
    CFG = json.load(fh)
RES = os.path.join(ROOT, CFG["outputs"]["results_dir"])
FIG = os.path.join(ROOT, CFG["outputs"]["figures_dir"])
LID = os.path.join(ROOT, CFG["outputs"]["lid_series_dir"])
for d in (RES, FIG):
    os.makedirs(d, exist_ok=True)

CONTROLS = CFG["study"]["controls"]
STYLE = {"uniform_E025": dict(color="#222222", ls="-", label="Uniform"),
         "B1_E025": dict(color="#1f77b4", ls="--", label=r"B1: $A_2\sin(2\pi x)$"),
         "B3_seed0_E025": dict(color="#d62728", ls="-.", label="B3 (seed 0)"),
         "B2_seed5_E025": dict(color="#2ca02c", ls=":", label="B2 (seed 5)")}


def load_scalars(tag):
    with open(os.path.join(RES, f"scalars_{tag}.json"), encoding="utf-8") as fh:
        return json.load(fh)


def load_fields(tag):
    with np.load(os.path.join(RES, f"fields_{tag}.npz")) as z:
        return {k: z[k] for k in z.files}


def load_lid(name):
    with np.load(os.path.join(LID, f"{name}_re500_e0.25.npz")) as z:
        if "U_series" in z:
            return z["U_series"], True
        return z["U_mean"][None, :], False


def lid_on_grid(name, x):
    ser, is_t = load_lid(name)
    x0 = np.linspace(0, 1, ser.shape[1])
    out = np.array([np.interp(x, x0, ser[t]) for t in range(ser.shape[0])])
    return out, is_t


def save(fig, name, dpi):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=dpi,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  figure {name}.pdf/png OK")


def style_axes(ax, is_geom):
    """Ticks exacts echelles geometrie (x,y in [0,1], aspect egal)."""
    if is_geom:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal")
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.tick_params(labelsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    # Détection automatique de la résolution à partir des scalars dispo
    import glob
    cand = [os.path.basename(p) for p in
            glob.glob(os.path.join(RES, "scalars_uniform_E025_Re500_N*.json"))]
    if not cand:
        N = CFG["lbm"]["N_default"]
    else:
        N = int(max(cand, key=lambda s: int(s.split("N")[1].split(".")[0]))
                .split("N")[1].split(".")[0])
    tags = {c: f"{c}_Re500_N{N}" for c in CONTROLS}

    scalars = {c: load_scalars(tags[c]) for c in CONTROLS}
    fields = {c: load_fields(tags[c]) for c in CONTROLS}
    x = np.linspace(0, 1, N); y = np.linspace(0, 1, N)

    # ========================================================= lid structure (1x4)
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.0), sharex=True,
                             constrained_layout=True)
    for ax, c in zip(axes, CONTROLS):
        ser, is_t = lid_on_grid(c, x)
        if is_t:
            nt = ser.shape[0]
            ph = np.linspace(0, nt - 1, 5).astype(int)
            for p in ph:
                ax.plot(x, ser[p], color="0.60", lw=.8, alpha=.55)
            ax.plot(x, ser.mean(axis=0), color=STYLE[c]["color"], lw=2.0,
                    ls="-", label="period mean")
            ax.legend(fontsize=6, loc="best", frameon=True)
        else:
            ax.plot(x, ser[0], color=STYLE[c]["color"], lw=2.0)
        ax.axhline(0, color="gray", lw=.5)
        ax.set_xlim(0, 1); ax.set_ylim(-1.35, 1.35)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.tick_params(labelsize=7)
        ax.set_title(STYLE[c]["label"], fontsize=9, pad=6)
        ax.set_xlabel(r"$x/L$", fontsize=8)
    axes[0].set_ylabel(r"$U_{lid}/U_{ref}$", fontsize=8)
    fig.suptitle(r"Lid controls — fixed actuation energy $E^*=0.25$, Re=500 (AR=1)",
                 fontsize=10)
    save(fig, "fig_lid_controls", args.dpi)

    # ===================================================== composite 4 x 3
    # (row0: U_lid — non-geometrique; rows 1-2: geometries exactes AR=1)
    fig = plt.figure(figsize=(15.6, 9.0), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.35, 1.35],
                          hspace=0.28, wspace=0.16)
    phis = {c: np.clip(fields[c]["phi"], 1e-14, None) for c in CONTROLS}
    phmin = min(p.min() for p in phis.values())
    phmax = max(p.max() for p in phis.values())
    umax = {c: float(np.hypot(fields[c]["ux"], fields[c]["uy"]).max())
            for c in CONTROLS}
    omax = {c: float(np.abs(fields[c]["vort"]).max()) for c in CONTROLS}

    for j, c in enumerate(CONTROLS):
        # ---- row 0 : U_lid (phases + mean for B2/B3) --------------------
        ax = fig.add_subplot(gs[0, j])
        ser, is_t = lid_on_grid(c, x)
        if is_t:
            nt = ser.shape[0]
            ph = np.linspace(0, nt - 1, 5).astype(int)
            for p in ph:
                ax.plot(x, ser[p], color="0.62", lw=.7, alpha=.6)
            ax.plot(x, ser.mean(axis=0), color=STYLE[c]["color"], lw=1.8,
                    label="mean")
            ax.legend(fontsize=6, loc="upper right", frameon=True)
        else:
            ax.plot(x, ser[0], color=STYLE[c]["color"], lw=1.8)
        ax.axhline(0, color="gray", lw=.4)
        ax.set_xlim(0, 1); ax.set_ylim(-1.4, 1.4)
        ax.set_xticks([0.0, 0.5, 1.0]); ax.tick_params(labelsize=6)
        ax.set_title(STYLE[c]["label"], fontsize=9, pad=5)
        if j == 0:
            ax.set_ylabel(r"$U_{lid}$", fontsize=8)
        else:
            ax.yaxis.set_visible(False)
        ax.set_xlabel("", fontsize=1)

        # ---- row 1 : streamlines + vorticity (exact AR=1) ----------------
        ax = fig.add_subplot(gs[1, j])
        vrt = fields[c]["vort"]
        sgn = np.sign(vrt); mx = float(np.abs(vrt).max())
        if mx > 0:
            levs = np.concatenate([np.linspace(-mx, -1e-4, 30),
                                   np.linspace(1e-4, mx, 30)])
        else:
            levs = 60 * [0.0]
        im = ax.contourf(x, y, vrt, levels=levs, cmap="RdBu_r", extend="both")
        ax.contour(x, y, fields[c]["psi"],
                   levels=np.linspace(fields[c]["psi"].min(),
                                      fields[c]["psi"].max(), 21),
                   colors="k", linewidths=0.35)
        style_axes(ax, is_geom=True)
        ax.set_xticks([0.0, 0.5, 1.0]); ax.set_yticks([0.0, 0.5, 1.0])
        ax.tick_params(labelsize=6)
        if j == 0:
            ax.set_ylabel(r"$y/L$", fontsize=8)
        ax.set_xlabel(r"$x/L$", fontsize=8)
        ax.text(0.02, 0.97, f"max|$\\omega$|={omax[c]:.2e}",
                transform=ax.transAxes, va="top", fontsize=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.6", alpha=0.8))
        ax.text(0.02, 0.75, rf"max$|u|$={umax[c]:.2e}",
                transform=ax.transAxes, va="top", fontsize=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.6", alpha=0.8))
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                          format=lambda v, _: f"{v:.0e}")
        cb.ax.tick_params(labelsize=6)
        cb.set_label(r"$\omega L/U_{ref}$", fontsize=7, labelpad=1)
        ax.set_title("")

        # ---- row 2 : log10 Phi(x,y) (exact AR=1, shared scale) -----------
        ax = fig.add_subplot(gs[2, j])
        pc = ax.pcolormesh(x, y, phis[c], cmap="inferno",
                           norm=LogNorm(vmin=phmin, vmax=phmax),
                           shading="auto")
        style_axes(ax, is_geom=True)
        ax.set_xticks([0.0, 0.5, 1.0]); ax.set_yticks([0.0, 0.5, 1.0])
        ax.tick_params(labelsize=6)
        if j == 0:
            ax.set_ylabel(r"$y/L$", fontsize=8)
        ax.set_xlabel(r"$x/L$", fontsize=8)
        ax.set_title("")

    # couleurbar partagée pour la ligne Phi (échelle log commune) uniquement ;
    # la vorticity a une colorbar par panneau (échelles propres par colonne).
    gs2 = gs[2, :]
    cax2 = fig.add_axes([gs2.get_position(fig).x0 + 0.05, gs2.get_position(fig).y0 - 0.12,
                         gs2.get_position(fig).width, 0.008])
    fig.colorbar(pc, cax=cax2, orientation="horizontal")
    cax2.set_title(r"$\Phi(x,y)$ (shared log10 scale, $2\nu\,S\!:\!S$)", fontsize=7)

    fig.suptitle((r"Fixed actuation energy $E^*=0.25$, Re=500, AR=1 — "
                  r"columns: control structure; rows: lid / streamlines + vorticity / "
                  r"$\log_{10}\Phi$"),
                 fontsize=10)
    save(fig, "fig_composite_4x3", args.dpi)

    # ======================================================= quantitative
    eps = np.array([scalars[c]["eps"] for c in CONTROLS])
    eps_u = eps[0]
    comp = np.array([[scalars[c][k] for c in CONTROLS]
                     for k in ("eps_xx", "eps_yy", "eps_xy")])
    peps = np.array([scalars[c]["P_lid"] / scalars[c]["eps"] for c in CONTROLS])
    rk = np.array([scalars[c]["R_K"] or 0.0 for c in CONTROLS])
    cols = [STYLE[c]["color"] for c in CONTROLS]
    labels = [STYLE[c]["label"].split(":")[0].split(" (")[0] for c in CONTROLS]
    xpos = np.arange(4)

    fig = plt.figure(figsize=(10.2, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(xpos, eps / eps_u, color=cols, edgecolor="k", linewidth=.7)
    ax.axhline(1.0, color="#222222", lw=1.0)
    ax.set_ylabel(r"$\varepsilon/\varepsilon_{uniform}$", fontsize=8)
    for i, v in enumerate(eps / eps_u):
        ax.text(i, v * 1.03, f"{v:.2f}", ha="center", va="bottom",
                color=cols[i], fontweight="bold", fontsize=8)
    ax.set_ylim(0, eps_u and max(1.55, (eps / eps_u).max() * 1.12))
    ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_title(r"(a) dissipation vs uniform", fontsize=9)

    ax = fig.add_subplot(gs[0, 1])
    bot = np.zeros(4)
    for compi, name, cc in zip(comp, ("xx", "yy", "xy"),
                               ("#4c72b0", "#dd8452", "#55a868")):
        ax.bar(xpos, compi, bottom=bot, color=[cc] * 4,
               edgecolor="k", linewidth=.4, label=rf"$\varepsilon_{{{name}}}$")
        bot = bot + compi
    ax.set_ylabel(r"$\varepsilon$", fontsize=8)
    ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=7)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.set_title(r"(b) strain-rate split "
                 r"$\varepsilon_{xx}+\varepsilon_{yy}+\varepsilon_{xy}$",
                 fontsize=9)

    ax = fig.add_subplot(gs[1, 0])
    ax.bar(xpos, peps, color=cols, edgecolor="k", linewidth=.7)
    ax.axhline(1.0, color="#222222", lw=1.0)
    ax.set_ylabel(r"$P_{lid}/\varepsilon$", fontsize=8)
    for i, v in enumerate(peps):
        ax.text(i, v * 1.03, f"{v:.2f}", ha="center",
                va="bottom", color=cols[i], fontsize=7)
    ax.set_ylim(0, 1.45)
    ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_title(r"(c) moving-wall work balance", fontsize=9)

    ax = fig.add_subplot(gs[1, 1])
    ax.bar(xpos, rk, color=cols, edgecolor="k", linewidth=.7)
    ax.set_ylabel(r"$R_K=K_{fluct}/K$", fontsize=8)
    for i, v in enumerate(rk):
        ax.text(i, v, f"{v:.2g}", ha="center", va="bottom", color=cols[i],
                fontsize=7)
    ax.set_yscale("symlog", linthresh=0.1, linscale=0.5)
    ax.set_yticks([0, 0.1, 1, 10])
    ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_title(r"(d) mean-flow unsteadiness flag (temporal controls)", fontsize=9)

    fig.suptitle(r"Fixed-energy quantitative comparison, Re=500 (solver N=%d)" % N,
                 fontsize=10)
    save(fig, "fig_quantitative", args.dpi)

    # ============================================================== tau_w
    fig, ax = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
    for c in CONTROLS:
        kw = {k: v for k, v in STYLE[c].items() if k != "label"}
        ax.plot(x, fields[c]["tau_w"], **kw, lw=1.6, label=STYLE[c]["label"])
    ax.axhline(0, color="gray", lw=0.7)
    ax.set_xlabel(r"$x/L$", fontsize=8)
    ax.set_ylabel(r"$\tau_w(x)/(\rho U_{ref}^2)$", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xlim(0, 1)
    ax.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left", fontsize=8)
    ax.set_title(r"Wall shear at the moving lid (y=1), AR=1", fontsize=9)
    save(fig, "fig_tau_w", args.dpi)

    # ============================================================ summary csv
    csvp = os.path.join(RES, "fig_summary_numbers.csv")
    with open(csvp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["control", "eps", "eps/eps_u", "frac_xx", "frac_yy",
                    "frac_xy", "K", "K_fluct", "R_K", "Z", "P_lid",
                    "P_lid/eps", "period", "stat_iters"])
        for c in CONTROLS:
            s = scalars[c]
            w.writerow([c, f"{s['eps']:.6e}", f"{s['eps']/eps_u:.6f}",
                        f"{s['frac_xx']:.4f}", f"{s['frac_yy']:.4f}",
                        f"{s['frac_xy']:.4f}", f"{s['K']:.6e}",
                        f"{s['K_fluct']:.6e}" if s.get('K_fluct') is not None else "",
                        f"{s['R_K']:.6f}" if s.get('R_K') is not None else "",
                        f"{s['Z']:.6e}", f"{s['P_lid']:.6e}",
                        f"{s['P_lid_balance']:.4f}", s['period'],
                        s['stat_iters']])
    print("[done] figures ->", FIG)
    print(f"  (N detecté = {N})")


if __name__ == "__main__":
    main()