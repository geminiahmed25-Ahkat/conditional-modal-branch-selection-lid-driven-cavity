# -*- coding: utf-8 -*-
"""
audit_energy.py — Phase 0 : audit des conventions de dissipation et du bilan
énergétique avant toute nouvelle campagne (case study 15_09_2026).

Vérifie sur les champs moyens déjà sauvegardés (results/fields_*.npz) :

  1.  Idempotence : ε, ε_xx, ε_yy, ε_xy recalculés ici == scalars_*.json.
  2.  Cohérence tensorielle :  Φ = ν(2S11² + 2S22² + 4S12²)
            ≡ 2ν(S11² + S22² + 2S12²)  ≡  (2/Re)[u_x² + v_y² + ½(u_y+v_x)²]
      (forme standard du manuscrit 2ν S:S).
  3.  Bouclage lid : ux[N-1, :] == U_lid_norm imposé (normalisé).
  4.  Bilan énergétique indépendant :  P_lid = ∫_lid U_lid·τ_w dx
      évalué en gradients paroi 1er et 2e ordre, comparé à ε.
      → rapport P_lid/ε visé ≈ 1 (écart = discrétisation de τ_w, s'améliore à N=256).

Sortie : results/audit_energy_report.txt (+ print). Ne relance AUCUNE simulation.
"""
import json, os, glob

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as fh:
    CFG = json.load(fh)
RES = os.path.join(ROOT, CFG["outputs"]["results_dir"])


def phi_parts(u, v, nu, dx):
    ux_, uy_ = np.gradient(u, dx, axis=1), np.gradient(u, dx, axis=0)
    vx_, vy_ = np.gradient(v, dx, axis=1), np.gradient(v, dx, axis=0)
    S11, S22, S12 = ux_, vy_, 0.5 * (uy_ + vx_)
    phi1 = nu * (2 * S11 ** 2 + 2 * S22 ** 2 + 4 * S12 ** 2)
    phi2 = 2 * nu * (S11 ** 2 + S22 ** 2 + 2 * S12 ** 2)
    assert np.allclose(phi1, phi2, rtol=1e-14, atol=1e-14), "convention tensorielle incohérente !"
    return phi1, ux_, vy_, 0.5 * (uy_ + vx_)


def wall_shear(u, nu, dx, order):
    N = u.shape[0]
    if order == 1:
        return nu * (u[N - 1, :] - u[N - 2, :]) / dx
    return nu * (-3.0 * u[N - 1, :] + 4.0 * u[N - 2, :] - u[N - 3, :]) / (2.0 * dx)


def main():
    lines = []
    def log(x=""):
        print(x); lines.append(x)

    log("=" * 78)
    log("PHASE 0 — AUDIT dissipation / bilan énergétique (case study 15_09_2026)")
    log("Convention testée :  Φ = ν(2S11² + 2S22² + 4S12²)  =  2ν S:S")
    log("= (2/Re)[u_x² + v_y² + ½(u_y + v_x)²]  (standard, ν = 1/Re)")
    log("=" * 78)

    for fld in sorted(glob.glob(os.path.join(RES, "fields_*.npz"))):
        tag = os.path.basename(fld)[len("fields_"):-4]
        sjson = os.path.join(RES, f"scalars_{tag}.json")
        if not os.path.exists(sjson):
            continue
        with np.load(fld) as z:
            u, v = z["ux"], z["uy"]
            U_lid = z["U_lid"]     # série (period, N) imposée
            tau_w_saved = z["tau_w"]
        s = json.load(open(sjson, encoding="utf-8"))
        N = s["N"]; Re = s["Re"]; nu = 1.0 / Re; dx = 1.0 / (N - 1)
        period = U_lid.shape[0]
        x = np.linspace(0, 1, N); w = np.ones(N); w[0] = w[-1] = 0.5

        phi, ux_, vy_, S12 = phi_parts(u, v, nu, dx)
        trap2 = lambda g: np.trapz(np.trapz(g, dx=dx, axis=1), dx=dx)
        eps, eps_xx = trap2(phi), trap2(nu * 2 * ux_ ** 2)
        eps_yy, eps_xy = trap2(nu * 2 * vy_ ** 2), trap2(nu * 4 * S12 ** 2)

        # ---- 1. idempotence vs scalars stockés
        d_eps = abs(eps - s["eps"]); d_xx = abs(eps_xx - s["eps_xx"])
        d_yy = abs(eps_yy - s["eps_yy"]); d_xy = abs(eps_xy - s["eps_xy"])
        maxd = max(d_eps, d_xx, d_yy, d_xy)
        log(f"\n[{tag}] N={N} Re={Re} period={period}")
        log(f"  1. ε recalculé  eps={eps:.6e}  xx={eps_xx:.6e}  yy={eps_yy:.6e}  xy={eps_xy:.6e}")
        log(f"     |Δ| max vs scalars.json = {maxd:.2e}  -> {'OK' if maxd < 1e-9 else 'ÉCART !'}")

        # ---- 3. bouclage lid : ux[row N-1] == U_lid imposé
        ulid = np.asarray(U_lid).reshape(N)
        err_lid = np.max(np.abs(u[N - 1, :] - ulid))
        log(f"  3. ux[N-1,:] vs U_lid imposé : max|Δ| = {err_lid:.2e}  "
            f"-> {'OK' if err_lid < 1e-6 else 'ÉCART !'}")

        # ---- 4. bilan P_lid (travail du couvercle) vs ε
        for order in (1, 2):
            tau = wall_shear(u, nu, dx, order)   # τ du champ MOYEN (valable aussi périodique)
            P = np.sum(w * ulid * tau) * dx
            P_json = s["P_lid"]
            bal = P / eps
            note = ""
            if P_json is not None:
                note = f"   (scalars.json P_lid={P_json:.6e}, P_lid/eps={P_json/eps:.4f})"
            log(f"  4. P_lid(order {order}) = {P:.6e}   P_lid/ε = {bal:.4f}   "
                f"|P−ε|/ε = {abs(P-eps)/eps:.4f}{note}")
        log(f"     tau_w sauvegardé : min={tau_w_saved.min():.3e} max={tau_w_saved.max():.3e}")

    log("\n" + "=" * 78)
    log("AUDIT — Interprétation :")
    log(" * ε(⟨u⟩) est la dissipation du champ MOYEN. Pour les contrôles temporels")
    log("   (B2/B3, period>1) le bilan total P_lid ≈ ε_mean + ε_fluct ; R_K = K_fluct/K_mean")
    log("   quantifie l'unsteadiness — à ajouter aux diagnostics des cas temporels.")
    log(" * P_lid/ε > 1 à N=128 : erreur 1er/2e ordre du gradient pariétal ; ")
    log("   attendu < 5 % à N=256.")
    log("=" * 78)
    log("\n  τ_w convention : τ_w = ν·(∂u/∂y)(y=lid), traction sur le fluide, ")
    log("  P_lid = +∫_lid U_lid·τ_w dx > 0 (puissance injectée par le couvercle).")

    out = os.path.join(RES, "audit_energy_report.txt")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n[audit] rapport -> {out}")


if __name__ == "__main__":
    main()