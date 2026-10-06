# -*- coding: utf-8 -*-
"""
run_all.py — Reproduction intégrale du case study Re=500, E*=0.25.

    py -3.11 run_all.py                 (toutes étapes, N=128, tol 1e-8, max_iter 1.2e6)
    py -3.11 run_all.py --N 256 --tol 1e-9 --max-iter 5000000   (config production)

Étapes :
  1. build_controls.py     → data/lid_series/*.npz + results/seed_controls_re500_e0.25.csv
  2. run_case_study.py     → results/* (champs + scalaires + summary)
  3. make_figures.py       → figures/*.pdf/png

Sort : la preuve de reproduction (checksums) dans logs/reproduction_checksums.txt.
"""
import argparse, csv, hashlib, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as fh:
    CFG = json.load(fh)
PY = sys.executable


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def run_step(script, *args):
    t0 = time.time()
    print(f"\n=== {script} {' '.join(map(str, args))} ===")
    r = subprocess.run([PY, os.path.join(HERE, script), *map(str, args)], cwd=HERE)
    if r.returncode != 0:
        sys.exit(f"ÉCHEC {script}")
    print(f"=== {script} OK ({time.time()-t0:.0f}s) ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=CFG["lbm"]["N_default"])
    ap.add_argument("--max-iter", type=int, default=CFG["lbm"]["max_iter_default"])
    ap.add_argument("--tol", type=float, default=CFG["lbm"]["tol_default"])
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-run", action="store_true")
    ap.add_argument("--skip-figures", action="store_true")
    args = ap.parse_args()

    if not args.skip_build:
        run_step("build_controls.py")
    if not args.skip_run:
        run_step("run_case_study.py", "--N", args.N, "--max-iter", args.max_iter,
                 "--tol", args.tol)
    if not args.skip_figures:
        run_step("make_figures.py", "--dpi", 150)

    # preuve de reproduction ------------------------------------------------------------------
    items = []
    for sub in ("results", "data"):
        for dp, dn, fn in os.walk(os.path.join(ROOT, sub)):
            for f in sorted(fn):
                items.append(os.path.relpath(os.path.join(dp, f), ROOT))
    lines = ["# Reproduction case study 15_09_2026 — checksums SHA256 (16 hex)",
             f"# généré le {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"# config : N={args.N} max_iter={args.max_iter} tol={args.tol}"]
    for rel in items:
        p = os.path.join(ROOT, rel)
        lines.append(f"{rel}\t{sha(p)}")
    out = os.path.join(ROOT, "logs", "reproduction_checksums.txt")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n[reproduction] checksums -> {out}")


if __name__ == "__main__":
    main()