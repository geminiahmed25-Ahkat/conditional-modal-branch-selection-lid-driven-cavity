# -*- coding: utf-8 -*-
"""
diagnostics.py — Diagnostics de mécanique des fluides calculés sur le champ LBM
convergé (moyenne temporelle, normalisé par U_ref) pour un contrôle donné.

Quantités calculées (convention non-dimensionnelle, nu = 1/Re, dx = 1/(N-1)) :
  - dissipation density   Phi(x,y) = nu*(2*S11**2 + 2*S22**2 + 4*S12**2)
                              S11=du/dx, S22=dv/dy, S12=0.5*(du/dy+dv/dx)
  - décomposition en strain : eps_xx, eps_yy, eps_xy (contributions à epsilon)
  - énergie cinétique K = ∫ 0.5 (u²+v²)
  - enstrophie        Z = ∫ vort²          (vort = dv/dx - du/dy)
  - fonction de courant ψ (intégration verticale de u en imposant ψ=0 en bas)
  - cisaillement pariétal au couvercle τ_w(x) = nu*(du/dy)(y=1) [signe convention]
  - travail du couvercle   P_lid = -∫_lid U_lid * τ_w dx   (bilan P_lid ≈ ε)
  - champ ε_xx, ε_yy, ε_xy en 2D (pour figures)

Entrée : objet LBM_MRT_Solver résolu (N, ux, uy, rho, U_lid_norm, Re).
Sortie : dict de scalaires + champs, et enregistrement NPZ reproductible.
"""
import numpy as np
import json, os


def compute(solver, tag, outdir):
    N = solver.N
    dx = 1.0 / (N - 1)
    nu = 1.0 / solver.Re
    U, V = solver.ux, solver.uy                       # normalisés par U_ref

    dUdx, dUdy = np.gradient(U, dx, axis=1), np.gradient(U, dx, axis=0)
    dVdx, dVdy = np.gradient(V, dx, axis=1), np.gradient(V, dx, axis=0)

    S11, S22, S12 = dUdx, dVdy, 0.5 * (dUdy + dVdx)
    phi = nu * (2 * S11 ** 2 + 2 * S22 ** 2 + 4 * S12 ** 2)

    # --- contributions par composante de strain ----------------------------
    phi_xx = nu * 2 * S11 ** 2
    phi_yy = nu * 2 * S22 ** 2
    phi_xy = nu * 4 * S12 ** 2
    trap2 = lambda g: np.trapz(np.trapz(g, dx=dx, axis=1), dx=dx)
    eps = trap2(phi); eps_xx = trap2(phi_xx); eps_yy = trap2(phi_yy); eps_xy = trap2(phi_xy)

    K = trap2(0.5 * (U ** 2 + V ** 2))
    vort = dVdx - dUdy
    Z = trap2(vort ** 2)

    # --- fonction de courant : ψ(x,y) = ∫_0^y u(x,s) ds, ψ(y=0)=0 ---------
    psi = np.cumsum(U, axis=0) * dx                     # (N, N), lignes = y croissant
    psi -= psi[0, :][None, :]                           # ψ(y=0) = 0

    # --- cisaillement pariétal et travail du couvercle ----------------------
    # τ_w = ν·(∂u/∂y)(y=lid), gradient 1er ordre sur le champ moyen.
    # P_lid est calculé avec la VITESSE RÉCUPÉRÉE u[N-1,:] (et non le profil
    # idéalisé) : le LBM satisfait uw au centre mais pas aux coins (singularité
    # de coin), d'où fermeture P_lid ≈ ε à ~1 % au lieu de >5 % (audit 15_09_2026).
    x = np.linspace(0, 1, N)
    du_dy_lid = (U[N - 1, :] - U[N - 2, :]) / dx        # dérivée 1er ordre au bord
    tau_w = nu * du_dy_lid                              # τ_w(x) (convention +, fluid-side)
    U_lid_rec = U[N - 1, :]
    U_lid_imposed = np.asarray(solver.U_lid_norm).reshape(solver.period, N)[0]
    period = int(solver.period)
    uw = np.ones(N); uw[0] = uw[-1] = 0.5
    P_lid = np.sum(uw * U_lid_rec * tau_w) * dx         # travail réellement injecté
    P_lid_balance = P_lid / eps

    out = {
        "control": tag, "Re": solver.Re, "N": N, "period": period,
        "iters": solver.iters,
        "last_residual": float(solver.last_residual), "periodic": bool(solver.periodic),
        "mean_count": int(solver.mean_count),
        "stat_iters": int(getattr(solver, "stat_iters", 0)),
        "eps": eps, "eps_xx": eps_xx, "eps_yy": eps_yy, "eps_xy": eps_xy,
        "frac_xx": eps_xx / eps, "frac_yy": eps_yy / eps, "frac_xy": eps_xy / eps,
        "K": K, "Z": Z, "P_lid": P_lid, "P_lid_balance": P_lid_balance,
        "K_fluct": float(solver.K_fluct) if hasattr(solver, "K_fluct") else None,
        "R_K": float(solver.K_fluct / K) if hasattr(solver, "K_fluct") else None,
        "eps_over_eps_uniform": None,  # rempli par run_case_study
        "x": x.tolist(), "y": np.linspace(0, 1, N).tolist(),
        "U_lid_mean": float(np.mean(U_lid_rec)),
    }
    fields = {"ux": U, "uy": V, "vort": vort, "phi": phi,
              "phi_xx": phi_xx, "phi_yy": phi_yy, "phi_xy": phi_xy, "psi": psi,
              "tau_w": tau_w, "U_lid": U_lid_rec, "U_lid_imposed": U_lid_imposed}
    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(os.path.join(outdir, f"fields_{tag}.npz"),
                        control=tag, Re=solver.Re, N=N, **fields)
    with open(os.path.join(outdir, f"scalars_{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out, fields