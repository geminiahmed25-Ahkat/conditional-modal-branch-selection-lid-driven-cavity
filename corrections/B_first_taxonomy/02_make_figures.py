# -*- coding: utf-8 -*-
"""
Paper 2 -- corrected figures with 4-class taxonomy (B_first).

Regenerates only the figures whose content changes when the first-harmonic
class B_first is introduced:
    fig03_lambda_branch_probabilities  (4 classes instead of 3)
    fig04_lambda_entropy               (entropy normalised by ln 4)
    fig07_loss_ablation                (adds f1 column; No-Lvar is B_first)
All other figures are unchanged because no B_first realization occurs in the
185-run reference campaign.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGET = r"C:\Users\user\Documents\travail pc taki_final\Default Project\papier 2 date_14_09_2026\Paper2_branch_selection\Paper2_clean_corrected"
OUT = os.path.join(TARGET, "03_figures")
os.makedirs(OUT, exist_ok=True)
DPI = 300
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "figure.dpi": 120, "savefig.bbox": "tight", "axes.grid": False})


def savefig(name):
    fig = plt.gcf()
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=DPI)
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    plt.tight_layout()
    plt.close()


# ---------------------------------------------------------------- lambda_var
LAMBDA_V = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 5.0])
CLASSES4 = ["B1", "B2", "B_first", "B3"]
# rows: lambda; cols: [B1, B2, B_first, B3]
LAMBDA_FREQ = {
    (500, 0.25): np.array([[0, 0, 1, 0],
                           [0.6, 0.4, 0, 0],
                           [0.4, 0.2, 0, 0.4],
                           [0.4, 0.2, 0, 0.4],
                           [0.4, 0.4, 0, 0.2],
                           [0.4, 0.6, 0, 0]]),
    (700, 2.0): np.array([[0, 0, 1, 0],
                          [0, 0, 1, 0],
                          [0, 0.6, 0, 0.4],
                          [0.2, 0.4, 0, 0.4],
                          [0.6, 0.2, 0, 0.2],
                          [0.6, 0.2, 0, 0.2]]),
    (1000, 1.0): np.array([[0, 0, 1, 0],
                           [0, 0.6, 0, 0.4],
                           [0.4, 0.4, 0, 0.2],
                           [0.4, 0.4, 0, 0.2],
                           [0.4, 0.4, 0, 0.2],
                           [0.4, 0.6, 0, 0]]),
}
LAMBDA_CELLS = [(500, 0.25), (700, 2.0), (1000, 1.0)]


def entropy4(P):
    P = np.asarray(P, dtype=float)
    out = []
    for p in P:
        pp = p[p > 0]
        out.append(-np.sum(pp * np.log(pp)) / np.log(4.0))
    return np.asarray(out)


# fig03 -- 4-class heatmaps
fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.6), sharey=True)
for ax, cell in zip(axes, LAMBDA_CELLS):
    F = LAMBDA_FREQ[cell]
    im = ax.imshow(F.T, origin="lower", aspect="auto", vmin=0, vmax=1,
                   extent=[LAMBDA_V[0], LAMBDA_V[-1], -0.5, 3.5], cmap="viridis")
    ax.set_title(rf"$Re={cell[0]},\ E^*={cell[1]:g}$")
    ax.set_xlabel(r"$\lambda_{\mathrm{var}}$")
    ax.set_xticks(LAMBDA_V)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["B1", "B2", "B$_{\\rm first}$", "B3"])
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            v = F[i, j]
            ax.text(LAMBDA_V[i], j, f"{v:.1f}", ha="center", va="center",
                    color="white" if v < 0.6 else "black", fontsize=8)
fig.colorbar(im, ax=axes.ravel().tolist(), label="Empirical probability")
fig.suptitle(r"Redistribution of branch-selection probability with $\lambda_{var}$ (4 classes)", y=1.02)
savefig("fig03_lambda_branch_probabilities")

# fig04 -- entropy (ln 4)
fig, ax = plt.subplots(figsize=(6.2, 4.2))
for cell in LAMBDA_CELLS:
    H = entropy4(LAMBDA_FREQ[cell])
    ax.plot(LAMBDA_V, H, marker="o", linewidth=1.6,
            label=rf"$Re={cell[0]},\ E^*={cell[1]:g}$")
ax.set_xlabel(r"$\lambda_{var}$")
ax.set_ylabel(r"Normalized branch-selection entropy $H_N$")
ax.set_ylim(-0.02, 1.02)
ax.set_xscale("symlog", linthresh=0.1)
ax.legend(frameon=False)
ax.set_title(r"Training-output diversity (entropy normalised over 4 classes)")
savefig("fig04_lambda_entropy")

# fig07 -- loss ablation with f1
ABLATION_LABELS = [
    "Baseline", "No $\\mathcal{L}_{var}$", "No $\\mathcal{L}_{Re}$",
    "No $\\mathcal{L}_{diss}$", "No $\\mathcal{L}_{var},\\mathcal{L}_{Re}$",
    "No $\\mathcal{L}_{var},\\mathcal{L}_{diss}$", "$\\lambda_{var}/2$",
    "$2\\lambda_{var}$"]
ABLATION_F1 = np.array([0.0, 90.4, 0.0, 0.0, 90.4, 90.5, 0.2, 0.1])
ABLATION_F2 = np.array([92.2, 0.2, 92.2, 92.3, 0.2, 0.3, 89.3, 92.8])
ABLATION_FT = np.array([1.3, 5.1, 1.3, 1.2, 5.1, 5.0, 1.5, 1.0])

fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0))
x = np.arange(len(ABLATION_LABELS))
for ax, vals, ylab in zip(axes, [ABLATION_F1, ABLATION_F2, ABLATION_FT],
                          [r"First-harmonic fraction $f_1$ (%)",
                           r"Second-harmonic fraction $f_2$ (%)",
                           r"Time-dependent fraction $f_t$ (%)"]):
    ax.bar(x, vals)
    ax.set_ylabel(ylab)
    ax.set_xticks(x)
    ax.set_xticklabels(ABLATION_LABELS, rotation=55, ha="right")
    ax.set_ylim(0, 100)
fig.suptitle(r"Loss-term ablation at $Re=500$: removing $\mathcal{L}_{var}$ selects the first-harmonic branch $B_{\rm first}$")
savefig("fig07_loss_ablation")

print("Corrected figures written to", OUT)
for n in ["fig03_lambda_branch_probabilities", "fig04_lambda_entropy", "fig07_loss_ablation"]:
    print("  ", n, os.path.exists(os.path.join(OUT, n + ".pdf")))
