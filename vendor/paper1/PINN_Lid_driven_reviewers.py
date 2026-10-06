# -*- coding: utf-8 -*-
"""
================================================================
PINN Lid-Driven Cavity — Reviewer Campaign Version
================================================================
Modified from historical baseline.  All reviewer-requested ablations,
sweeps, and sensitivity studies are implemented here.

Change MODE at the top to select which campaign to run.
================================================================
"""

import sys

# Force UTF-8 output to avoid UnicodeEncodeError on cp1252 consoles
# (must be done before any print with non-ASCII chars like lambda)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ─── Imports ────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad as torch_grad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pysr import PySRRegressor
import os, time, json, csv, warnings, random, gc
warnings.filterwarnings('ignore')

try:
    import pandas as pd
except ImportError:
    pd = None

# ─── MODE selector ──────────────────────────────────────────────
MODE = "seed_study"
# MODE = "baseline"         # reproduction historique
# MODE = "loss_ablation"     # R2.2 / R3.2
# MODE = "energy_sweep"      # R3.2
# MODE = "seed_study"        # R2.4 / R3.3
# MODE = "architecture"      # R2.4
# MODE = "sampling"          # R2.4
# MODE = "mode_count"        # R2.4 / R3.2
# MODE = "temporal"          # R2.4 / R3.2
# MODE = "aspect_ratio"      # R3.2
# MODE = "unseen_re"         # robustesse inter-Re
# MODE = "symbolic"          # R2.8 / R3.1
# MODE = "convergence"       # R2.7
# MODE = "parametrization"   # R2-2 (2 familles de base)
# MODE = "all"               # toutes les campagnes

# ─── Seed utility ───────────────────────────────────────────────
SEEDS = list(range(10))

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ─── Loss ablation flags ────────────────────────────────────────
USE_PDE  = True
USE_BC   = True
USE_DISS = True
USE_CTRL = True
USE_VAR  = True
USE_RE   = True

# ─── Hyperparamètres (corrigés) ────────────────────────────────
LX       = 1.0
LY       = 1.0
T_MAX    = 1.0
RE_LIST  = [100, 500, 1000]
Re_min, Re_max = 100.0, 1000.0

N_EPOCHS_PRE  = 300
N_EPOCHS_MAIN = 3000
N_PHYS  = 4000
N_BC    = 600
LR      = 8e-4
N_MX    = 6
N_MT    = 5

LAMBDA_PDE   = 1.0
LAMBDA_BC    = 15.0
LAMBDA_DISS  = 0.05
LAMBDA_CTRL  = 200.0
LAMBDA_VAR   = 1.0
LAMBDA_RE    = 2.0
E_TARGET     = 0.25

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT_DIR", os.path.join(SCRIPT_DIR, "results_reviewers"))
os.makedirs(OUT, exist_ok=True)

# ─── Architecture configs ───────────────────────────────────────
ARCHITECTURES = {
    "small": {
        "net": [64, 64, 32],
        "coeff": [32, 32]
    },
    "baseline": {
        "net": [128, 128, 96, 64],
        "coeff": [64, 64, 48]
    },
    "large": {
        "net": [256, 256, 128, 96],
        "coeff": [128, 128, 64]
    }
}

# ─── Sampling configs ──────────────────────────────────────────
SAMPLES = {
    "low":      (2000, 300),
    "baseline": (4000, 600),
    "high":     (8000, 1200)
}

# ─── Mode configs (number of modes) ────────────────────────────
MODE_CONFIGS = [
    (4, 3),
    (6, 1),
    (6, 3),
    (6, 5),
    (8, 5),
    (8, 9),
]

# ─── Temporal configs ──────────────────────────────────────────
TEMPORAL_CONFIGS = [1, 3, 5]

# ─── Energy sweep ──────────────────────────────────────────────
ENERGY_TARGETS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00]

# ─── Unseen Re ─────────────────────────────────────────────────
UNSEEN_RE = [200, 350, 650, 800, 950]

# ─── Device ─────────────────────────────────────────────────────
DTYPE  = torch.float64
torch.set_default_dtype(DTYPE)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {device}")

# ─── Normalisation Re ──────────────────────────────────────────
def normalize_Re(Re):
    return 2.0 * (Re - Re_min) / (Re_max - Re_min) - 1.0

# ─── Architecture ──────────────────────────────────────────────
class SineAct(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1, dtype=DTYPE))
    def forward(self, x):
        return torch.sin(self.a * x)


class UltraPINN(nn.Module):
    """
    PINN unifié (x,y,t,Re) avec loi de contrôle U_lid(x,t,Re).
    Architecture paramétrable via arch_config.
    basis_type : "fourier" (sin((i+1)pi x/Lx)) ou
                 "chebyshev_mod" (x(Lx-x) * T_i(2x/Lx-1)) qui satisfait
                 U(0)=U(Lx)=0, au sens lisse (non sinusoïdal).
    """
    def __init__(self, arch_config=None, n_mx=None, n_mt=None, lx=None, ly=None,
                 basis_type="fourier"):
        super().__init__()
        self.n_mx = n_mx if n_mx is not None else N_MX
        self.n_mt = n_mt if n_mt is not None else N_MT
        self.lx   = lx   if lx   is not None else LX
        self.ly   = ly   if ly   is not None else LY
        self.basis_type = basis_type

        if arch_config is None:
            arch_config = ARCHITECTURES["baseline"]

        net_dims  = arch_config["net"]
        coeff_dims = arch_config["coeff"]

        def blk(i, o):
            return nn.Sequential(nn.Linear(i, o, dtype=DTYPE), SineAct())

        layers = []
        in_dim = 4
        for h in net_dims:
            layers.append(blk(in_dim, h))
            in_dim = h
        layers.append(nn.Linear(in_dim, 3, dtype=DTYPE))
        self.net = nn.Sequential(*layers)

        n = self.n_mx * self.n_mt
        clayers = []
        cin = 1
        for h in coeff_dims:
            clayers.append(blk(cin, h))
            cin = h
        clayers.append(nn.Linear(cin, n, dtype=DTYPE))
        self.coeff_net = nn.Sequential(*clayers)

        with torch.no_grad():
            bias = self.coeff_net[-1].bias
            bias.zero_()
            nmt = self.n_mt

            def mode_index(i, j):
                return i * nmt + j

            if self.n_mx >= 1 and self.n_mt >= 1:
                bias[mode_index(0, 0)] = 0.50
            if self.n_mx >= 1 and self.n_mt >= 2:
                bias[mode_index(0, 1)] = 0.18
            if self.n_mx >= 2 and self.n_mt >= 1:
                bias[mode_index(1, 0)] = 0.20
            if self.n_mx >= 2 and self.n_mt >= 2:
                bias[mode_index(1, 1)] = 0.08

    def U_lid(self, x, t, Re_phys):
        xn = x / self.lx
        tn = t / T_MAX
        Re_n = normalize_Re(Re_phys).view(-1, 1)
        basis = []

        if self.basis_type == "fourier":
            for i in range(self.n_mx):
                for j in range(self.n_mt):
                    f = torch.sin((i+1) * np.pi * xn) * torch.cos(j * np.pi * tn)
                    basis.append(f)
        elif self.basis_type == "chebyshev_mod":
            # Base lisse satisfaisant U(0)=U(Lx)=0 :
            #   phi_i(x) = x*(Lx-x) * T_i(xi),  xi = 2x/Lx - 1 in [-1,1]
            # T_i = Chebyshev (premiere espece). T_0=1, T_1=xi, T_2=2xi^2-1, ...
            x_phys = x.detach() if not x.requires_grad else x
            xi = 2.0 * (x_phys / self.lx) - 1.0
            envelope = x_phys * (self.lx - x_phys)
            for i in range(self.n_mx):
                # Chebyshev T_i via recurrence
                if i == 0:
                    T_i = torch.ones_like(xi)
                elif i == 1:
                    T_i = xi
                else:
                    T_im2 = torch.ones_like(xi)
                    T_im1 = xi
                    T_i = None
                    for _ in range(2, i+1):
                        T_i = 2.0 * xi * T_im1 - T_im2
                        T_im2 = T_im1
                        T_im1 = T_i
                s = envelope * T_i
                for j in range(self.n_mt):
                    f = s * torch.cos(j * np.pi * tn)
                    basis.append(f)
        else:
            raise ValueError(f"Unknown basis_type: {self.basis_type}")

        basis = torch.cat(basis, dim=1)
        coeffs = self.coeff_net(Re_n)
        U_phys = (coeffs * basis).sum(dim=1, keepdim=True)
        return U_phys

    def forward(self, x_n, y_n, t_n, Re_n):
        inp = torch.cat([x_n, y_n, t_n, Re_n], dim=1)
        out = self.net(inp)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]

    def net_params(self):
        return list(self.net.parameters())

    def ctrl_params(self):
        return list(self.coeff_net.parameters())


# ─── Dérivées ──────────────────────────────────────────────────
def D(u, v, cg=True):
    return torch_grad(u, v, grad_outputs=torch.ones_like(u),
                      create_graph=cg, retain_graph=True)[0]

def lap(u, x, y):
    return D(D(u, x), x) + D(D(u, y), y)


# ─── Génération des points ─────────────────────────────────────
def gen_points(Re_phys_val, n_phys=None, n_bc=None, lx=None, ly=None):
    _n_phys = n_phys if n_phys is not None else N_PHYS
    _n_bc   = n_bc   if n_bc   is not None else N_BC
    _lx     = lx     if lx     is not None else LX
    _ly     = ly     if ly     is not None else LY

    xp = torch.rand(_n_phys, 1, dtype=DTYPE) * _lx
    yp = torch.rand(_n_phys, 1, dtype=DTYPE) * _ly
    tp = torch.rand(_n_phys, 1, dtype=DTYPE) * T_MAX
    rp = torch.full((_n_phys, 1), Re_phys_val, dtype=DTYPE)

    def wall(n, xv=None, yv=None):
        if xv is not None:
            X = torch.full((n, 1), xv, dtype=DTYPE)
            Y = torch.rand(n, 1, dtype=DTYPE) * _ly
        else:
            X = torch.rand(n, 1, dtype=DTYPE) * _lx
            Y = torch.full((n, 1), yv, dtype=DTYPE)
        T = torch.rand(n, 1, dtype=DTYPE) * T_MAX
        R = torch.full((n, 1), Re_phys_val, dtype=DTYPE)
        return X, Y, T, R

    bc = {
        'bottom': wall(_n_bc, yv=0.),
        'top'   : wall(_n_bc, yv=_ly),
        'left'  : wall(_n_bc, xv=0.),
        'right' : wall(_n_bc, xv=_lx),
    }
    for k in bc:
        bc[k] = tuple(v.to(device) for v in bc[k])

    xp = xp.to(device).requires_grad_(True)
    yp = yp.to(device).requires_grad_(True)
    tp = tp.to(device).requires_grad_(True)
    rp = rp.to(device).requires_grad_(False)
    return (xp, yp, tp, rp), bc


# ─── Fonction de coût ──────────────────────────────────────────
def compute_loss(model, pts, bc, nu, Re_val, ctrl_active=True):
    xp, yp, tp, rp = pts
    _lx = model.lx
    _ly = model.ly

    x_n = 2.0 * (xp - 0.0) / _lx - 1.0
    y_n = 2.0 * (yp - 0.0) / _ly - 1.0
    t_n = 2.0 * (tp - 0.0) / T_MAX - 1.0
    r_n = normalize_Re(rp)

    u_n, v_n, p_n = model(x_n, y_n, t_n, r_n)
    u = u_n * 1.0
    v = v_n * 1.0
    p = p_n * 1.0

    res_u = D(u, tp) + u*D(u, xp) + v*D(u, yp) + D(p, xp) - nu*lap(u, xp, yp)
    res_v = D(v, tp) + u*D(v, xp) + v*D(v, yp) + D(p, yp) - nu*lap(v, xp, yp)
    res_c = D(u, xp) + D(v, yp)
    l_pde = (res_u**2 + res_v**2 + res_c**2).mean()

    l_bc = torch.tensor(0., dtype=DTYPE, device=device)
    for w in ['bottom', 'left', 'right']:
        xb, yb, tb, rb = bc[w]
        xb_n = 2.0 * (xb - 0.0) / _lx - 1.0
        yb_n = 2.0 * (yb - 0.0) / _ly - 1.0
        tb_n = 2.0 * (tb - 0.0) / T_MAX - 1.0
        rb_n = normalize_Re(rb)
        ub_n, vb_n, _ = model(xb_n, yb_n, tb_n, rb_n)
        ub = ub_n * 1.0
        vb = vb_n * 1.0
        l_bc += ub.pow(2).mean() + vb.pow(2).mean()

    xb, yb, tb, rb = bc['top']
    xb_n = 2.0 * (xb - 0.0) / _lx - 1.0
    yb_n = 2.0 * (yb - 0.0) / _ly - 1.0
    tb_n = 2.0 * (tb - 0.0) / T_MAX - 1.0
    rb_n = normalize_Re(rb)
    ub_n, vb_n, _ = model(xb_n, yb_n, tb_n, rb_n)
    ub = ub_n * 1.0
    vb = vb_n * 1.0
    u_lid = model.U_lid(xb, tb, rb)
    l_bc += (ub - u_lid).pow(2).mean() + vb.pow(2).mean()

    ux = D(u, xp); uy = D(u, yp)
    vx = D(v, xp); vy = D(v, yp)
    l_diss = nu * (ux**2 + vy**2 + 2*(0.5*(uy+vx))**2).mean()

    res_u_abs = res_u.detach().abs()
    res_v_abs = res_v.detach().abs()
    res_c_abs = res_c.detach().abs()
    residual_stats = {
        "PDE_MSE": float(l_pde.item()),
        "BC_MSE": float(l_bc.item()),
        "dissipation": float(l_diss.item()),
        "Ru_mean": float(res_u_abs.mean().item()),
        "Ru_max":  float(res_u_abs.max().item()),
        "Rv_mean": float(res_v_abs.mean().item()),
        "Rv_max":  float(res_v_abs.max().item()),
        "Rc_mean": float(res_c_abs.mean().item()),
        "Rc_max":  float(res_c_abs.max().item()),
    }

    if not ctrl_active:
        total = LAMBDA_PDE * l_pde + LAMBDA_BC * l_bc
        return total, l_pde.item(), l_bc.item(), 0., 0., 0., 0., residual_stats

    energy = (u_lid**2).mean()
    l_ctrl = (energy - E_TARGET)**2

    u_mean = u_lid.mean()
    u_var  = ((u_lid - u_mean)**2).mean()
    l_var  = -torch.log(u_var + 1e-5)

    Re_low  = torch.tensor([[Re_min]], dtype=DTYPE, device=device)
    Re_high = torch.tensor([[Re_max]], dtype=DTYPE, device=device)
    x_mid = torch.linspace(0, _lx, 20, dtype=DTYPE, device=device).view(-1,1)
    t_mid = torch.full_like(x_mid, 0.5)
    U_low  = model.U_lid(x_mid, t_mid, Re_low.expand(20,1))
    U_high = model.U_lid(x_mid, t_mid, Re_high.expand(20,1))
    diff_Re = ((U_low - U_high)**2).mean()
    l_re = torch.relu(0.01 - diff_Re)

    total = (
        (LAMBDA_PDE  * l_pde  if USE_PDE  else 0.0) +
        (LAMBDA_BC   * l_bc   if USE_BC   else 0.0) +
        (LAMBDA_DISS * l_diss if USE_DISS else 0.0) +
        (LAMBDA_CTRL * l_ctrl if USE_CTRL else 0.0) +
        (LAMBDA_VAR  * l_var  if USE_VAR  else 0.0) +
        (LAMBDA_RE   * l_re   if USE_RE   else 0.0)
    )

    return (total, l_pde.item(), l_bc.item(), l_diss.item(),
            l_ctrl.item(), energy.item(), u_var.item(), residual_stats)


# ─── Entraînement ──────────────────────────────────────────────
def train_ultra(verbose=True, n_mx=None, n_mt=None, arch_config=None,
                n_phys=None, n_bc=None, e_target=None, lx=None, ly=None,
                re_list=None, lambdas_override=None, use_flags_override=None,
                out_dir=None, collect_residuals=False, basis_type=None,):
    global N_MX, N_MT, N_PHYS, N_BC, E_TARGET, LX, LY
    global USE_PDE, USE_BC, USE_DISS, USE_CTRL, USE_VAR, USE_RE
    global LAMBDA_PDE, LAMBDA_BC, LAMBDA_DISS, LAMBDA_CTRL, LAMBDA_VAR, LAMBDA_RE
    global RE_LIST

    _saved = {}
    _overrides = {
        'N_MX': n_mx, 'N_MT': n_mt, 'N_PHYS': n_phys, 'N_BC': n_bc,
        'E_TARGET': e_target, 'LX': lx, 'LY': ly,
    }
    for gname, val in _overrides.items():
        if val is not None:
            _saved[gname] = globals()[gname]
            globals()[gname] = val

    if re_list is not None:
        _saved['RE_LIST'] = globals()['RE_LIST']
        globals()['RE_LIST'] = list(re_list)

    if lambdas_override:
        for key, val in lambdas_override.items():
            gname = f"LAMBDA_{key}"
            if gname in globals():
                _saved[gname] = globals()[gname]
                globals()[gname] = val

    if use_flags_override:
        for key, val in use_flags_override.items():
            gname = f"USE_{key}"
            if gname in globals():
                _saved[gname] = globals()[gname]
                globals()[gname] = val

    try:
        return _train_ultra_impl(verbose=verbose, arch_config=arch_config,
                                 out_dir=out_dir,
                                 collect_residuals=collect_residuals,
                                 basis_type=basis_type)
    finally:
        for gname, val in _saved.items():
            globals()[gname] = val


def _train_ultra_impl(verbose=True, arch_config=None, out_dir=None,
                      collect_residuals=False, basis_type=None):
    _n_mx  = N_MX
    _n_mt  = N_MT
    _lx    = LX
    _ly    = LY
    _e_tgt = E_TARGET
    _n_re  = len(RE_LIST)
    if basis_type is None:
        basis_type = "fourier"

    print(f"\n{'='*65}")
    print(f"ULTRA PINN (corrigé) — Entraînement unifié Re={RE_LIST}")
    print(f"Pré-entraînement : {N_EPOCHS_PRE} ép.  Principal : {N_EPOCHS_MAIN} ép.")
    print(f"λ_ctrl={LAMBDA_CTRL} (symétrique)  λ_var={LAMBDA_VAR}  λ_re={LAMBDA_RE}  E_target={_e_tgt}")
    print(f"Modes : {_n_mx}×{_n_mt}={_n_mx*_n_mt}  N_PHYS={N_PHYS}  N_BC={N_BC}")
    print(f"Basis : {basis_type}")
    print(f"{'='*65}")

    model = UltraPINN(arch_config=arch_config, n_mx=_n_mx, n_mt=_n_mt,
                      lx=_lx, ly=_ly, basis_type=basis_type).to(device)
    t0    = time.time()

    datasets = {}
    for Re in RE_LIST:
        pts, bc = gen_points(Re, lx=_lx, ly=_ly)
        datasets[Re] = (pts, bc, 1.0/Re, Re)

    print(f"\n[PHASE 1] Pré-entraînement {N_EPOCHS_PRE} ép. — coeff_net gelé")
    opt_pre = optim.Adam(model.net_params(), lr=LR)
    for ep in range(1, N_EPOCHS_PRE + 1):
        opt_pre.zero_grad()
        total = torch.tensor(0., dtype=DTYPE, device=device)
        for Re, (pts, bc, nu, re_val) in datasets.items():
            loss, *_ = compute_loss(model, pts, bc, nu, re_val, ctrl_active=False)
            total += loss
        if torch.isnan(total):
            print(f"  NaN @ {ep}"); break
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.net_params(), 1.0)
        opt_pre.step()
        if verbose and ep % 100 == 0:
            print(f"  Ep {ep:4d}/{N_EPOCHS_PRE} | Loss={total.item():.3e}")
    print(f"  Pre-training done {time.time()-t0:.0f}s")

    print(f"\n[PHASE 2] Entraînement principal {N_EPOCHS_MAIN} ép. — tous paramètres")
    opt = optim.Adam(model.parameters(), lr=LR)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS_MAIN, eta_min=5e-6)

    hist = {k: [] for k in ['total', 'pde', 'bc', 'diss', 'ctrl', 'energy', 'var']}
    residual_history = []
    t1   = time.time()

    for ep in range(1, N_EPOCHS_MAIN + 1):
        opt.zero_grad()
        total = torch.tensor(0., dtype=DTYPE, device=device)
        sums  = [0.] * 6

        ep_residual = None
        for Re, (pts, bc, nu, re_val) in datasets.items():
            out = compute_loss(model, pts, bc, nu, re_val, ctrl_active=True)
            loss = out[0]
            total += loss
            for i in range(6):
                sums[i] += out[i + 1]
            ep_residual = out[7]

        if torch.isnan(total):
            print(f"  NaN @ epoque {ep}"); break
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()

        hist['total'].append(total.item())
        hist['pde'].append(sums[0]);    hist['bc'].append(sums[1])
        hist['diss'].append(sums[2]);   hist['ctrl'].append(sums[3])
        hist['energy'].append(sums[4]); hist['var'].append(sums[5])

        if collect_residuals and ep_residual is not None:
            residual_history.append(ep_residual)

        if verbose and (ep % 300 == 0 or ep == 1):
            elapsed = time.time() - t1
            eta     = elapsed / ep * (N_EPOCHS_MAIN - ep)
            print(f"  Ep {ep:4d}/{N_EPOCHS_MAIN} | Loss={total.item():.3e} | "
                  f"PDE={sums[0]:.3e} | E={sums[4]/_n_re:.4f} | "
                  f"Var={sums[5]/_n_re:.4f} | ETA={eta/60:.1f}min")

    print(f"\n  Training done {(time.time()-t0)/60:.1f} min | "
          f"Loss={hist['total'][-1]:.4e} | E_mean={hist['energy'][-1]/_n_re:.4f}")

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        hist_df = pd.DataFrame(hist) if pd is not None else None
        if hist_df is not None:
            hist_df['epoch'] = range(1, len(hist_df)+1)
            hist_df.to_csv(os.path.join(out_dir, "training_history.csv"), index=False)
        if collect_residuals and residual_history:
            rdf = pd.DataFrame(residual_history) if pd is not None else None
            if rdf is not None:
                rdf.to_csv(os.path.join(out_dir, "residual_stats.csv"), index=False)
        torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

    return model, hist, residual_history


# ─── Extraction U_lid ──────────────────────────────────────────
def extract_lid(model, nx=80, nt=80, lx=None, re_list=None):
    _lx = lx if lx is not None else LX
    _re_list = re_list if re_list is not None else RE_LIST
    rows = []
    row_dict = {}
    for Re in _re_list:
        xs = torch.linspace(0, _lx, nx, dtype=DTYPE, device=device).view(-1, 1)
        ts = torch.linspace(0, T_MAX, nt, dtype=DTYPE, device=device).view(-1, 1)
        X, T = torch.meshgrid(xs[:,0], ts[:,0], indexing='ij')
        Xf, Tf = X.reshape(-1,1), T.reshape(-1,1)
        Re_tensor = torch.full_like(Xf, Re)
        with torch.no_grad():
            U = model.U_lid(Xf, Tf, Re_tensor).cpu().numpy()
        data_re = np.column_stack([
            Xf.cpu().numpy(), Tf.cpu().numpy(),
            np.full((len(U),1), Re), U
        ])
        rows.append(data_re)
        row_dict[Re] = data_re
    return np.vstack(rows), row_dict


# ─── Analyse de contrôle (modal) ───────────────────────────────
def analyze_control(model, Re, lx=None, ly=None, nx=200, nt=200):
    lx = lx or LX
    ly = ly or LY
    n_mx = model.n_mx
    n_mt = model.n_mt

    xs = torch.linspace(0, lx, nx, dtype=DTYPE, device=device).view(-1, 1)
    ts = torch.linspace(0, T_MAX, nt, dtype=DTYPE, device=device).view(-1, 1)
    X, T = torch.meshgrid(xs[:,0], ts[:,0], indexing='ij')
    Xf, Tf = X.reshape(-1,1), T.reshape(-1,1)
    Re_tensor = torch.full_like(Xf, Re)

    with torch.no_grad():
        U = model.U_lid(Xf, Tf, Re_tensor).cpu().numpy().ravel()

    U_grid = U.reshape(nx, nt)

    x_arr = xs.cpu().numpy().ravel()
    t_arr = ts.cpu().numpy().ravel()

    dx = lx / (nx - 1)
    dt = T_MAX / (nt - 1)

    # Surface du domaine spatio-temporel (normalisation énergie)
    domain_area = lx * T_MAX

    # Poids de quadrature trapézoïdale standard (bords à demi-poids) en 2D.
    # Toutes les intégrales (normes modales, projections, énergie totale) utilisent
    # la MÊME forme bilinéaire, de sorte que modal_coverage -> 1.0 exact lorsqu'une
    # fonction est exactement dans la base (cohérent, contrairement à np.mean + somme
    # de trapèzes aux bords sur-comptés qui donnait ~1.01 pour le mode constant j=0).
    wx = np.ones(nx); wx[0] = 0.5; wx[-1] = 0.5
    wt = np.ones(nt); wt[0] = 0.5; wt[-1] = 0.5
    W = dx * dt * np.outer(wx, wt)          # poids 2D (nx, nt), somme = domain_area

    # Énergie totale normalisée : moyenne pondérée (quadrature trapézoïdale) spatio-temporelle
    E_total = np.sum(W * (U_grid ** 2)) / domain_area

    # Coefficients NATIFS de la parametrisation (base du modele, Fourier ou
    # Chebyshev modifie). On les conserve SEPAREMENT de la projection Fourier,
    # car A20/mode2_fraction/Efrac reposent sur la projection Fourier commune.
    native_coeffs = {}
    with torch.no_grad():
        Re_native = normalize_Re(torch.tensor([[Re]], dtype=DTYPE, device=device)).view(-1, 1)
        c_native = model.coeff_net(Re_native).squeeze().cpu().numpy()
    n_modes_total = n_mx * n_mt
    if c_native.size < n_modes_total:
        c_native = np.pad(c_native, (0, n_modes_total - c_native.size), mode='constant')
    for idx in range(n_modes_total):
        i_n = idx // n_mt
        j_n = idx % n_mt
        native_coeffs[f"native_c_{i_n}_{j_n}"] = float(c_native[idx])

    mode_energies = {}
    coefficients = {}
    for i in range(n_mx):
        for j in range(n_mt):
            si = np.sin((i+1)*np.pi*x_arr/lx)
            cj = np.cos(j*np.pi*t_arr)
            outer = np.outer(si, cj)
            norm2 = np.sum(W * (outer**2)) + 1e-30
            coeff = np.sum(W * U_grid * outer) / norm2
            # Énergie modale normalisée : E_ij = c_ij^2 * ||phi||^2 / aire
            energy_ij = coeff**2 * norm2 / domain_area
            mode_energies[(i, j)] = float(energy_ij)
            coefficients[f"c_{i}_{j}"] = float(coeff)

    dominant = max(mode_energies, key=mode_energies.get)

    # Somme des énergies des projections Fourier retenues sur la base tronquée
    # (pas nécessairement egale a E_total si la base est tronquee ou si le
    #  controle n'est pas exactement dans la famille Fourier)
    E_modes_sum = sum(mode_energies.values())

    # Couverture modale : fraction de E_total capturee par les modes retenus
    # Pour une bonne reconstruction, modal_coverage ~ 1
    modal_coverage = E_modes_sum / (E_total + 1e-30)

    E_mode2 = mode_energies.get((1, 0), 0.0)
    frac_mode2 = E_mode2 / (E_total + 1e-30)

    E_temporal = sum(v for (ii, jj), v in mode_energies.items() if jj > 0)
    temporal_frac = E_temporal / (E_total + 1e-30)

    recon = np.zeros_like(U_grid)
    for (i, j) in mode_energies:
        si = np.sin((i+1)*np.pi*x_arr/lx)
        cj = np.cos(j*np.pi*t_arr)
        outer = np.outer(si, cj)
        norm2 = np.sum(W * (outer**2)) + 1e-30
        coeff = np.sum(W * U_grid * outer) / norm2
        recon += coeff * outer
    recon_rmse = np.sqrt(np.mean((U_grid - recon)**2))

    # Amplitude du mode stationnaire (i,j=0) : A_i0 = c_{i,0}
    # NOTE : A20 = c_{1,0} est l'amplitude du mode sin(2*pi*x/Lx)*1
    # (mode spatial i=1, mode temporel j=0 = stationnaire)
    result = {
        "Re": Re,
        "E_total": float(E_total),
        "E_modes_sum": float(E_modes_sum),
        "modal_coverage": float(modal_coverage),
        "domain_area": float(domain_area),
        "A10": coefficients.get("c_0_0", 0.0),
        "A20": coefficients.get("c_1_0", 0.0),
        "A30": coefficients.get("c_2_0", 0.0),
        "A40": coefficients.get("c_3_0", 0.0),
        "A50": coefficients.get("c_4_0", 0.0),
        "A60": coefficients.get("c_5_0", 0.0),
        "A70": coefficients.get("c_6_0", 0.0),
        "A80": coefficients.get("c_7_0", 0.0),
        "mode2_fraction": float(frac_mode2),
        "fourier_mode2_fraction": float(frac_mode2),
        "temporal_fraction": float(temporal_frac),
        "dominant_i": int(dominant[0]),
        "dominant_j": int(dominant[1]),
        "reconstruction_rmse": float(recon_rmse),
    }
    # Distribution énergétique modale normalisée : fractions E_ij / E_total
    for (i, j), e_ij in mode_energies.items():
        result[f"Efrac_{i}_{j}"] = e_ij / (E_total + 1e-30)
    result.update(coefficients)
    # Coefficients natifs de la parametrisation (separes de la projection Fourier)
    result.update(native_coeffs)
    result["basis_type"] = str(getattr(model, "basis_type", "fourier"))
    return result


# ─── Régression symbolique avec PySR ──────────────────────────
def run_pysr(data, label="", niter=500):
    X = data[:, :3].astype(np.float32)
    y = data[:, 3].astype(np.float32)

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std

    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-12:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    print(f"\n  [PySR] {label} — {len(X)} points, vars=(x,t,Re_norm)")
    model_sr = PySRRegressor(
        niterations=niter,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp", "log", "square", "sqrt"],
        loss="L2DistLoss()",
        populations=50,
        maxsize=15,
        constraints={"/": (2, 1), "sin": 1, "cos": 1, "exp": 1, "log": 1, "sqrt": 1},
        parsimony=0.01,
        random_state=42,
        deterministic=True,
        parallelism="serial",
        verbosity=1
    )

    model_sr.fit(X_norm, y_norm)
    eq_norm = model_sr.sympy()

    import sympy as sp
    x_sym, t_sym, Re_sym = sp.symbols('x t Re')
    from sympy import symbols as syms
    x0, x1, x2 = syms('x0 x1 x2')
    expr = sp.sympify(str(eq_norm))
    expr = expr.subs(x0, (x_sym - float(X_mean[0])) / float(X_std[0]))
    expr = expr.subs(x1, (t_sym - float(X_mean[1])) / float(X_std[1]))
    expr = expr.subs(x2, (Re_sym - float(X_mean[2])) / float(X_std[2]))
    expr_final = sp.simplify(float(y_mean) + float(y_std) * expr)

    f = sp.lambdify((x_sym, t_sym, Re_sym), expr_final, modules='numpy')
    y_pred = f(data[:, 0], data[:, 1], data[:, 2])
    mse = np.mean((data[:, 3] - y_pred)**2)
    mae = np.mean(np.abs(data[:, 3] - y_pred))
    rmse = np.sqrt(mse)
    r2 = 1 - mse / np.var(data[:, 3])
    relative_L2 = np.sqrt(mse) / (np.std(data[:, 3]) + 1e-30)
    max_abs_error = np.max(np.abs(data[:, 3] - y_pred))

    print(f"  MSE = {mse:.4e}, MAE = {mae:.4e}, RMSE = {rmse:.4e}, R² = {r2:.4f}")
    print(f"  relative_L2 = {relative_L2:.4e}, max_abs_error = {max_abs_error:.4e}")
    print(f"  Equation: {expr_final}")
    return (model_sr, str(expr_final), mse, r2, mae, rmse, relative_L2, max_abs_error)


# ─── Validation LOCO ───────────────────────────────────────────
def loco_ultra(all_data_by_re, re_list=None):
    _re_list = re_list if re_list is not None else RE_LIST
    print(f"\n{'='*65}\nVALIDATION LOCO — ULTRA\n{'='*65}")
    results = {}
    for test_Re in _re_list:
        train_data = np.vstack([
            all_data_by_re[Re] for Re in _re_list if Re != test_Re
        ])
        _, expr, mse_tr, r2_tr, mae_tr, rmse_tr, rel2_tr, maxe_tr = \
            run_pysr(train_data, f"train excl Re={test_Re}", niter=200)

        td = all_data_by_re[test_Re]
        import sympy as sp
        f = sp.lambdify((sp.Symbol('x'), sp.Symbol('t'), sp.Symbol('Re')), expr, modules='numpy')
        y_pred = f(td[:, 0], td[:, 1], td[:, 2])
        y_true = td[:, 3]
        mse = float(((y_true - y_pred)**2).mean())
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(mse))
        r2 = float(1 - mse / y_true.var())
        rel2 = float(np.sqrt(mse) / (np.std(y_true) + 1e-30))
        maxe = float(np.max(np.abs(y_true - y_pred)))
        print(f"  LOCO Re={test_Re} → MSE={mse:.4e}  R²={r2:.4f}  MAE={mae:.4e}")
        results[test_Re] = {
            'expr': expr, 'mse_test': mse, 'r2_test': r2,
            'mae_test': mae, 'rmse_test': rmse, 'relative_L2_test': rel2,
            'max_abs_error_test': maxe,
            'mse_train': mse_tr, 'r2_train': r2_tr,
            'data': td
        }
    return results


# ─── Figures publication ───────────────────────────────────────
def make_all_figures(model, hist, loco_results, all_data_by_re, out_dir=None):
    if out_dir is None:
        out_dir = OUT
    os.makedirs(out_dir, exist_ok=True)
    _lx = LX
    _n_re = len(RE_LIST)

    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 13,
        'axes.labelsize': 15, 'axes.titlesize': 15,
        'legend.fontsize': 11, 'figure.dpi': 300, 'savefig.dpi': 300,
        'lines.linewidth': 1.8,
        'xtick.labelsize': 13, 'ytick.labelsize': 13,
    })
    colors = {'100': '#1D9E75', '500': '#185FA5', '1000': '#D85A30'}
    re_mk = {100: 'o', 500: 's', 1000: '^'}
    re_col = {100: 'k', 500: '0.45', 1000: '0.7'}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    ax = axes[0]
    ax.semilogy(hist['total'], color='#2C2C2A', lw=2, label='Total')
    ax.semilogy(hist['pde'],   color='#185FA5', ls='--', label='NS PDE')
    ax.semilogy(hist['bc'],    color='#D85A30', ls=':', label='BC')
    ax.set_xlabel('Epoch (–)'); ax.set_ylabel('Loss (–, log)')
    ax.set_title('Global convergence')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy(hist['ctrl'], color='#D85A30', label='Control energy')
    ax.semilogy(hist['diss'], color='#888780', ls='--', label='Dissipation (–)')
    ax2 = ax.twinx()
    e_arr = np.array(hist['energy']) / _n_re
    ax2.plot(e_arr, color='#BA7517', lw=1.3, alpha=0.85, label='Mean energy (–)')
    ax2.axhline(E_TARGET, color='#BA7517', ls='--', lw=0.8, alpha=0.5)
    ax2.set_ylabel('Energy (–)', color='#BA7517')
    ax.set_xlabel('Epoch (–)'); ax.set_ylabel('Loss (–, log)')
    ax.set_title('Control-energy terms')
    ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)

    ax = axes[2]
    v_arr = np.array(hist['var']) / _n_re
    ax.plot(v_arr, color='#1D9E75', label='Var $U_{lid}$ (–)')
    ax.axhline(0, color='gray', lw=0.7, ls=':')
    ax.set_xlabel('Epoch (–)'); ax.set_ylabel('Spatial variance (–)')
    ax.set_title('Spatial variance of $U_{lid}$ (–)')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.suptitle('Ultra PINN — full convergence', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig1_convergence.png"), dpi=300, bbox_inches='tight')
    plt.close(); print("  -> fig1_convergence")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    x_grid = np.linspace(0, _lx, 300)
    Xp = torch.tensor(x_grid[:, None], dtype=DTYPE, device=device)
    for ax, Re in zip(axes, RE_LIST):
        for t_val, ls, alpha in zip(
            [0., 0.2, 0.4, 0.6, 0.8, 1.0],
            ['-', '--', '-.', ':', '-', '--'],
            [1.0, 0.85, 0.7, 0.7, 0.85, 1.0]
        ):
            Tp = torch.full_like(Xp, t_val)
            Re_t = torch.full_like(Xp, Re)
            with torch.no_grad():
                Up = model.U_lid(Xp, Tp, Re_t).cpu().numpy().ravel()
            ax.plot(x_grid, Up, ls=ls, alpha=alpha,
                    label=f't={t_val:.1f}', color=re_col[Re], marker=re_mk[Re], markevery=30)
        ax.axhline(0, color='gray', lw=0.6, ls=':')
        ax.set_xlabel('$x$ (–)'); ax.set_ylabel('$U_{lid}(x,t)$ (–)')
        ax.set_title(f'Re = {Re}')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.suptitle('Ultra PINN — spatio-temporal lid profiles $U_{lid}(x,t)$', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_lid_profiles.png"), dpi=300, bbox_inches='tight')
    plt.close(); print("  -> fig2_lid_profiles")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    xs = np.linspace(0, _lx, 100); ts = np.linspace(0, T_MAX, 100)
    Xg, Tg = np.meshgrid(xs, ts)
    Xf = torch.tensor(Xg.ravel()[:, None], dtype=DTYPE, device=device)
    Tf = torch.tensor(Tg.ravel()[:, None], dtype=DTYPE, device=device)
    vmax_global = 0.
    for Re in RE_LIST:
        Re_t = torch.full_like(Xf, Re)
        with torch.no_grad():
            Uf = model.U_lid(Xf, Tf, Re_t).cpu().numpy().reshape(100, 100)
        vmax_global = max(vmax_global, np.abs(Uf).max())
    for ax, Re in zip(axes, RE_LIST):
        Re_t = torch.full_like(Xf, Re)
        with torch.no_grad():
            Uf = model.U_lid(Xf, Tf, Re_t).cpu().numpy().reshape(100, 100)
        im = ax.contourf(xs, ts, Uf, levels=40, cmap='RdBu_r',
                          vmin=-vmax_global, vmax=vmax_global)
        plt.colorbar(im, ax=ax, label='$U_{lid}$ (–)')
        ax.set_xlabel('$x$ (–)'); ax.set_ylabel('$t$ (–)')
        ax.set_title(f'Re = {Re}')
    fig.suptitle('Ultra PINN — $U_{lid}(x,t)$ heatmap per Re', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close(); print("  -> fig3_heatmap")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    Re_scan = np.array([50, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000])
    x_mids = [0.25, 0.5, 0.75]
    ax = axes[0]
    for x_mid in x_mids:
        U_scan = []
        for Re_v in Re_scan:
            Xp_ = torch.tensor([[x_mid]], dtype=DTYPE, device=device)
            Tp_ = torch.tensor([[0.5]], dtype=DTYPE, device=device)
            Re_t = torch.tensor([[Re_v]], dtype=DTYPE, device=device)
            with torch.no_grad():
                U_scan.append(model.U_lid(Xp_, Tp_, Re_t).item())
        ax.plot(Re_scan, U_scan, 'o-', lw=1.8, ms=5, label=f'x={x_mid:.2f}')
    ax.set_xlabel('$Re$ (–)'); ax.set_ylabel('$U_{lid}(x, t=0.5)$ (–)')
    ax.set_title('Identified law vs Re (linear)')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    U_mid = []
    for Re_v in Re_scan:
        Xp_ = torch.tensor([[0.5]], dtype=DTYPE, device=device)
        Tp_ = torch.tensor([[0.5]], dtype=DTYPE, device=device)
        Re_t = torch.tensor([[Re_v]], dtype=DTYPE, device=device)
        with torch.no_grad():
            U_mid.append(model.U_lid(Xp_, Tp_, Re_t).item())
    U_mid = np.array(U_mid)
    mask = U_mid > 0
    if mask.sum() > 2:
        ax.loglog(Re_scan[mask], U_mid[mask], 'ko-', ms=6, label='$U_{lid}$ (x=0.5, t=0.5)')
        log_re = np.log(Re_scan[mask].astype(float))
        log_u  = np.log(U_mid[mask])
        alpha_pow = np.polyfit(log_re, log_u, 1)[0]
        C = np.exp(np.polyfit(log_re, log_u, 1)[1])
        Re_fit = np.linspace(50, 2000, 200)
        ax.loglog(Re_fit, C * Re_fit**alpha_pow, '--', color='0.4',
                  lw=1.8, label=f'$U \\sim Re^{{{alpha_pow:.2f}}}$')
        ax.legend()
    ax.set_xlabel('$Re$ (–, log)'); ax.set_ylabel('$U_{lid}$ (–, log)')
    ax.set_title('Power law $U_{lid} \\sim Re^{\\alpha}$')
    ax.grid(True, which='both', alpha=0.3)
    fig.suptitle('Ultra PINN — Reynolds dependence', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig4_Re_law.png"), dpi=300, bbox_inches='tight')
    plt.close(); print("  -> fig4_Re_law")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for Re in RE_LIST:
        Re_t = torch.tensor([[Re]], dtype=DTYPE, device=device)
        with torch.no_grad():
            coeffs = model.coeff_net(normalize_Re(Re_t).view(-1,1)).squeeze().cpu().numpy()
        ax = axes[0]
        ax.plot(range(len(coeffs)), coeffs, 'o-', lw=1.5, ms=4,
                label=f'Re={Re}', color=re_col[Re], marker=re_mk[Re])
    axes[0].axhline(0, color='gray', lw=0.7, ls=':')
    axes[0].set_xlabel('Modal index (–)')
    axes[0].set_ylabel('Coefficient (–)'); axes[0].set_title('Modal coefficients vs Re')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    ax = axes[1]
    Re_scan_fine = np.linspace(50, 1500, 80)
    coeff_matrix = []
    for Re_v in Re_scan_fine:
        Re_t = torch.tensor([[Re_v]], dtype=DTYPE, device=device)
        with torch.no_grad():
            c = model.coeff_net(normalize_Re(Re_t).view(-1,1)).squeeze().cpu().numpy()
        coeff_matrix.append(c)
    coeff_matrix = np.array(coeff_matrix)
    n_modes = model.n_mx * model.n_mt
    im = ax.imshow(coeff_matrix.T, aspect='auto', cmap='RdBu_r',
                    origin='lower',
                    extent=[Re_scan_fine[0], Re_scan_fine[-1], 0, n_modes])
    plt.colorbar(im, ax=ax, label='Coefficient (–)')
    ax.set_xlabel('$Re$ (–)'); ax.set_ylabel('Modal index (–)')
    ax.set_title('Coefficient map vs Re')
    fig.suptitle('Ultra PINN — modal coefficients $U_{lid}$', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig5_coefficients.png"), dpi=300, bbox_inches='tight')
    plt.close(); print("  -> fig5_coefficients")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, test_Re in zip(axes, RE_LIST):
        res = loco_results[test_Re]
        td  = res['data']
        import sympy as sp
        f = sp.lambdify((sp.Symbol('x'), sp.Symbol('t'), sp.Symbol('Re')), res['expr'], modules='numpy')
        y_pred = f(td[:, 0], td[:, 1], td[:, 2])
        y_true = td[:, 3]
        ax.scatter(y_true, y_pred, s=3, alpha=0.4, color=re_col[test_Re], marker=re_mk[test_Re])
        m = min(y_true.min(), y_pred.min()); M = max(y_true.max(), y_pred.max())
        ax.plot([m, M], [m, M], 'k--', lw=1)
        ax.set_xlabel('$U_{lid}$ PINN (–)'); ax.set_ylabel('$U_{lid}$ SR (–)')
        ax.set_title(f'LOCO Re={test_Re}  R²={res["r2_test"]:.4f}')
        expr_s = res['expr'][:42] + '...' if len(res['expr']) > 42 else res['expr']
        ax.annotate(f'SR: {expr_s}', xy=(0.02, 0.03),
                    xycoords='axes fraction', fontsize=6.5, color='#333')
        ax.grid(True, alpha=0.3)
    fig.suptitle('Ultra PINN — LOCO validation', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig6_loco.png"), dpi=300, bbox_inches='tight')
    plt.close(); print("  -> fig6_loco")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    n = 30
    xs = np.linspace(0, _lx, n); ys = np.linspace(0, LY, n)
    Xg, Yg = np.meshgrid(xs, ys)
    Xf = torch.tensor(Xg.ravel()[:, None], dtype=DTYPE, device=device)
    Yf = torch.tensor(Yg.ravel()[:, None], dtype=DTYPE, device=device)
    Tf = torch.full_like(Xf, 0.5)
    for ax, Re in zip(axes, RE_LIST):
        Re_t = torch.full_like(Xf, Re)
        Xn = 2.0 * (Xf - 0.0) / _lx - 1.0
        Yn = 2.0 * (Yf - 0.0) / LY - 1.0
        Tn = 2.0 * (Tf - 0.0) / T_MAX - 1.0
        Rn = normalize_Re(Re_t)
        with torch.no_grad():
            u_n, v_n, _ = model(Xn, Yn, Tn, Rn)
        u = u_n.cpu().numpy().reshape(n, n)
        v = v_n.cpu().numpy().reshape(n, n)
        spd = np.sqrt(u**2 + v**2)
        strm = ax.streamplot(xs, ys, u, v, color=spd,
                              cmap='Greys', density=1.4, linewidth=0.8)
        plt.colorbar(strm.lines, ax=ax, label='$|u|$ (–)')
        ax.set_title(f'Re={Re}  (t=0.5)')
        ax.set_xlabel('$x$ (–)'); ax.set_ylabel('$y$ (–)')
    fig.suptitle('Ultra PINN — velocity fields under PINN-identified control',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig7_velocity.png"), dpi=300, bbox_inches='tight')
    plt.close(); print("  -> fig7_velocity")

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis('off')
    rows_t = [['Re (–)', 'MSE LOCO', 'R² LOCO', 'SR expression']]
    for Re in RE_LIST:
        res = loco_results[Re]
        expr = res['expr']
        rows_t.append([
            str(Re),
            f"{res['mse_test']:.3e}",
            f"{res['r2_test']:.4f}",
            expr[:80] + ('...' if len(expr) > 80 else '')
        ])
    tbl = ax.table(cellText=rows_t[1:], colLabels=rows_t[0],
                    cellLoc='left', loc='center',
                    bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#ccc')
        if r == 0:
            cell.set_facecolor('#e8f0fe')
            cell.set_text_props(fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#f8f8f8')
        if c == 3:
            cell.set_width(0.5)
    ax.set_title('Ultra PINN — symbolic-regression expressions (LOCO)', fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig8_sr_table.png"), dpi=300, bbox_inches='tight')
    plt.close(); print("  -> fig8_sr_table")

    fig = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.42, wspace=0.35,
                             left=0.05, right=0.97, top=0.92, bottom=0.08)
    ax = fig.add_subplot(gs[0, 0])
    ax.semilogy(hist['total'], color='k', lw=1.5, label='Total')
    ax.semilogy(hist['pde'],   color='0.45', ls='--', lw=1, label='PDE')
    ax.set_xlabel('Epoch (–)'); ax.set_ylabel('Loss (–)'); ax.set_title('Convergence')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(np.array(hist['energy'])/_n_re, color='k', lw=1.5)
    ax.axhline(E_TARGET, color='0.5', ls='--', lw=1, alpha=0.8, label=f'E_target={E_TARGET}')
    ax.set_xlabel('Epoch (–)'); ax.set_ylabel('Mean energy (–)'); ax.set_title('Energy convergence')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    x_grid_plot = np.linspace(0, _lx, 300)
    Xp_plot = torch.tensor(x_grid_plot[:, None], dtype=DTYPE, device=device)
    for col, Re in zip([2, 3], [100, 1000]):
        ax = fig.add_subplot(gs[0, col])
        for t_val, ls in zip([0., 0.25, 0.5, 0.75, 1.], ['-','--','-.',':','-']):
            Tp_plot = torch.full_like(Xp_plot, t_val)
            Re_t_plot = torch.full_like(Xp_plot, Re)
            with torch.no_grad():
                Up_plot = model.U_lid(Xp_plot, Tp_plot, Re_t_plot).cpu().numpy().ravel()
            ax.plot(x_grid_plot, Up_plot, ls=ls, alpha=0.8, label=f't={t_val:.2f}',
                    color=re_col[Re], marker=re_mk[Re], markevery=40)
        ax.axhline(0, color='gray', lw=0.6, ls=':')
        ax.set_xlabel('$x$ (–)'); ax.set_ylabel('$U_{lid}$ (–)'); ax.set_title(f'Re={Re}')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    for col, test_Re in enumerate(RE_LIST):
        ax = fig.add_subplot(gs[1, col])
        res = loco_results[test_Re]
        td  = res['data']
        import sympy as sp
        f = sp.lambdify((sp.Symbol('x'), sp.Symbol('t'), sp.Symbol('Re')), res['expr'], modules='numpy')
        yp_ = f(td[:, 0], td[:, 1], td[:, 2])
        yt_ = td[:, 3]
        ax.scatter(yt_, yp_, s=3, alpha=0.4, color=re_col[test_Re], marker=re_mk[test_Re])
        m_ = min(yt_.min(), yp_.min()); M_ = max(yt_.max(), yp_.max())
        ax.plot([m_, M_], [m_, M_], 'k--', lw=0.8)
        ax.set_title(f'LOCO Re={test_Re}  R²={res["r2_test"]:.3f}')
        ax.set_xlabel('$U_{lid}$ PINN (–)'); ax.set_ylabel('$U_{lid}$ SR (–)')
        ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 3])
    if mask.sum() > 2:
        ax.loglog(Re_scan[mask], U_mid[mask], 'ko-', lw=1.8, ms=5)
        ax.loglog(Re_fit, C * Re_fit**alpha_pow, '--', color='0.45',
                  lw=1.5, label=f'$Re^{{{alpha_pow:.2f}}}$')
        ax.legend(fontsize=8)
    ax.set_xlabel('$Re$ (–)'); ax.set_ylabel('$U_{lid}(0.5, 0.5)$ (–)')
    ax.set_title('Power law in Re'); ax.grid(True, which='both', alpha=0.3)

    fig.suptitle(
        'Ultra PINN — active control of driven cavity\n'
        'Universal lid law $U_{lid}(x,t,Re)$ + symbolic regression',
        fontsize=13, fontweight='bold'
    )
    plt.savefig(os.path.join(out_dir, "fig9_synthesis.png"), dpi=300, bbox_inches='tight')
    plt.close(); print("  -> fig9_synthesis")


# ─── Metadata helper ───────────────────────────────────────────
def save_metadata(out_dir, extra=None):
    metadata = {
        "energy_definition": "mean(U_lid**2)",
        "E_target": E_TARGET,
        "N_MX": N_MX,
        "N_MT": N_MT,
        "LX": LX,
        "LY": LY,
        "Re_min": Re_min,
        "Re_max": Re_max,
        "LAMBDA_PDE": LAMBDA_PDE,
        "LAMBDA_BC": LAMBDA_BC,
        "LAMBDA_DISS": LAMBDA_DISS,
        "LAMBDA_CTRL": LAMBDA_CTRL,
        "LAMBDA_VAR": LAMBDA_VAR,
        "LAMBDA_RE": LAMBDA_RE,
        "N_PHYS": N_PHYS,
        "N_BC": N_BC,
    }
    if extra:
        metadata.update(extra)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


# =================================================================
# CAMPAGNE 01 : BASELINE (reproduction exacte)
# =================================================================
def run_baseline():
    print("\n" + "=" * 65)
    print("CAMPAGNE 01 : BASELINE — reproduction historique")
    print("=" * 65)

    set_seed(42)
    out_dir = os.path.join(OUT, "01_baseline")
    os.makedirs(out_dir, exist_ok=True)

    t_global = time.time()

    model, hist, _ = train_ultra()

    print(f"\n{'='*65}\nEXTRACTION U_lid\n{'='*65}")
    all_data_flat, all_data_by_re = extract_lid(model, nx=80, nt=80)
    np.savetxt(os.path.join(out_dir, "lid_all.txt"), all_data_flat,
               header="x t Re_norm U_lid")
    for Re in RE_LIST:
        np.savetxt(os.path.join(out_dir, f"lid_Re{Re}.txt"), all_data_by_re[Re],
                   header="x t Re_norm U_lid")
    print(f"  {len(all_data_flat)} points extraits")

    loco_results = loco_ultra(all_data_by_re)

    print(f"\n{'='*65}\nGENERATION DES FIGURES\n{'='*65}")
    make_all_figures(model, hist, loco_results, all_data_by_re, out_dir=out_dir)

    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    print(f"  -> model.pt")

    metrics = {
        str(Re): {
            'loco_mse':   loco_results[Re]['mse_test'],
            'loco_r2':    loco_results[Re]['r2_test'],
            'loco_expr':  loco_results[Re]['expr'],
            'energy_final': hist['energy'][-1] / len(RE_LIST),
        } for Re in RE_LIST
    }
    metrics['_meta'] = {
        'N_EPOCHS_PRE': N_EPOCHS_PRE, 'N_EPOCHS_MAIN': N_EPOCHS_MAIN,
        'N_PHYS': N_PHYS, 'N_BC': N_BC,
        'LAMBDA_CTRL': LAMBDA_CTRL, 'LAMBDA_VAR': LAMBDA_VAR,
        'E_TARGET': E_TARGET,
        'total_time_min': (time.time() - t_global) / 60
    }
    with open(os.path.join(out_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 65)
    print("RESULTATS FINAUX — BASELINE")
    print(f"{'Re':>6} | {'Energie':>8} | {'LOCO R2':>8} | Expression SR")
    print("-" * 65)
    for Re in RE_LIST:
        res = loco_results[Re]
        expr_s = res['expr'][:45] + '...' if len(res['expr']) > 45 else res['expr']
        print(f"{Re:>6} | {hist['energy'][-1]/len(RE_LIST):>8.4f} | "
              f"{res['r2_test']:>8.4f} | {expr_s}")
    print(f"\n  Duree totale : {(time.time()-t_global)/60:.1f} min")
    print(f"  Fichiers dans : {out_dir}/")
    print("=" * 65)

    save_metadata(out_dir, extra={"campaign": "baseline", "seed": 42})
    return model, hist, loco_results


# =================================================================
# CAMPAGNE 02 : LOSS ABLATION
# =================================================================
def run_loss_ablation():
    print("\n" + "=" * 65)
    print("CAMPAGNE 02 : LOSS ABLATION — Re=500")
    print("=" * 65)

    out_dir = os.path.join(OUT, "02_loss_ablation")
    os.makedirs(out_dir, exist_ok=True)
    _save_re = list(RE_LIST)

    ablation_configs = {
        "BASELINE":       {},
        "NO_LVAR":        {"use_flags": {"VAR": False}},
        "NO_LRE":         {"use_flags": {"RE": False}},
        "NO_LDISS":       {"use_flags": {"DISS": False}},
        "NO_LVAR_LRE":    {"use_flags": {"VAR": False, "RE": False}},
        "NO_LVAR_LDISS":  {"use_flags": {"VAR": False, "DISS": False}},
        "LVAR_HALF":      {"lambdas": {"VAR": LAMBDA_VAR * 0.5}},
        "LVAR_DOUBLE":    {"lambdas": {"VAR": LAMBDA_VAR * 2.0}},
        "LDISS_HALF":     {"lambdas": {"DISS": LAMBDA_DISS * 0.5}},
        "LDISS_DOUBLE":   {"lambdas": {"DISS": LAMBDA_DISS * 2.0}},
        "LRE_HALF":       {"lambdas": {"RE": LAMBDA_RE * 0.5}},
        "LRE_DOUBLE":     {"lambdas": {"RE": LAMBDA_RE * 2.0}},
        "LCTRL_HALF":     {"lambdas": {"CTRL": LAMBDA_CTRL * 0.5}},
        "LCTRL_DOUBLE":   {"lambdas": {"CTRL": LAMBDA_CTRL * 2.0}},
    }

    results_rows = []

    for name, cfg in ablation_configs.items():
        print(f"\n--- Ablation: {name} ---")
        set_seed(42)
        sub_dir = os.path.join(out_dir, name)

        lambdas_ov = cfg.get("lambdas", None)
        use_ov     = cfg.get("use_flags", None)

        model, hist, _ = train_ultra(
            re_list=[500],
            lambdas_override=lambdas_ov,
            use_flags_override=use_ov,
            out_dir=sub_dir,
        )

        ac = analyze_control(model, Re=500)
        ac["config"] = name
        ac["energy_final"] = ac["E_total"]
        results_rows.append(ac)

        save_metadata(sub_dir, extra={"campaign": "loss_ablation", "config": name, "seed": 42})

        print(f"  Config={name} | E={ac['E_total']:.4f} | A1={ac['A10']:.4f} | "
              f"A2={ac['A20']:.4f} | mode2_frac={ac['mode2_fraction']:.4f}")

    if pd is not None:
        df = pd.DataFrame(results_rows)
        df.to_csv(os.path.join(out_dir, "loss_ablation_results.csv"), index=False)
    else:
        with open(os.path.join(out_dir, "loss_ablation_results.csv"), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results_rows[0].keys())
            writer.writeheader()
            writer.writerows(results_rows)

    print(f"\n  Resultats sauvegardes dans {out_dir}/loss_ablation_results.csv")
    return results_rows


# =================================================================
# CAMPAGNE 03 : ENERGY SWEEP
# =================================================================
def run_energy_sweep():
    print("\n" + "=" * 65)
    print("CAMPAGNE 03 : ENERGY SWEEP — Re=500")
    print("=" * 65)

    out_dir = os.path.join(OUT, "03_energy_sweep")
    os.makedirs(out_dir, exist_ok=True)

    results_rows = []

    for e_tgt in ENERGY_TARGETS:
        print(f"\n--- Energy target: {e_tgt} ---")
        set_seed(42)
        sub_dir = os.path.join(out_dir, f"E{e_tgt:.2f}")
        os.makedirs(sub_dir, exist_ok=True)

        model_path = os.path.join(sub_dir, "model.pt")

        if os.path.exists(model_path):
            print(f"  [SKIP] model.pt already exists for E_target={e_tgt}, loading...")
            model = UltraPINN(lx=LX, ly=LY)
            model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
            model.eval()
        else:
            model, hist, _ = train_ultra(
                re_list=[500],
                e_target=e_tgt,
                out_dir=sub_dir,
            )

        ac = analyze_control(model, Re=500)
        ac["E_target"] = e_tgt
        ac["energy_final"] = ac["E_total"]
        results_rows.append(ac)

        save_metadata(sub_dir, extra={"campaign": "energy_sweep", "E_target": e_tgt, "seed": 42})

        print(f"  E_target={e_tgt:.2f} | E_actual={ac['E_total']:.4f} | "
              f"A1={ac['A10']:.4f} | A2={ac['A20']:.4f} | mode2_frac={ac['mode2_fraction']:.4f}")

        # Free memory between trainings to avoid OOM on limited-RAM machines
        del model
        if 'hist' in locals():
            del hist
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if pd is not None:
        df = pd.DataFrame(results_rows)
        df.to_csv(os.path.join(out_dir, "energy_sweep.csv"), index=False)
    else:
        with open(os.path.join(out_dir, "energy_sweep.csv"), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results_rows[0].keys())
            writer.writeheader()
            writer.writerows(results_rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    e_targets = [r["E_target"] for r in results_rows]
    m2_fracs  = [r["mode2_fraction"] for r in results_rows]
    a2_vals   = [r["A20"] for r in results_rows]

    axes[0].plot(e_targets, m2_fracs, 'ko-', lw=2, ms=6)
    axes[0].set_xlabel('$E_{target}$ (–)'); axes[0].set_ylabel('Mode-2 fraction (–)')
    axes[0].set_title('Mode-2 fraction vs energy target'); axes[0].grid(True, alpha=0.3)

    axes[1].plot(e_targets, a2_vals, 'ks-', lw=2, ms=6)
    axes[1].set_xlabel('$E_{target}$ (–)'); axes[1].set_ylabel('$A_2$ (–)')
    axes[1].set_title('$A_2$ vs energy target'); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "energy_sweep_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n  Results saved in {out_dir}/energy_sweep.csv")
    return results_rows


# =================================================================
# CAMPAGNE 04 : SEED STUDY
# =================================================================
def run_seed_study():
    print("\n" + "=" * 65)
    print("CAMPAGNE 04 : SEED STUDY — Re=500")
    print("=" * 65)

    out_dir = os.path.join(OUT, "04_seed_study")
    os.makedirs(out_dir, exist_ok=True)

    results_rows = []

    for seed in SEEDS:
        print(f"\n--- Seed: {seed} ---")
        set_seed(seed)
        sub_dir = os.path.join(out_dir, f"seed_{seed}")

        model, hist, _ = train_ultra(
            re_list=[500],
            out_dir=sub_dir,
        )

        ac = analyze_control(model, Re=500)
        ac["seed"] = seed
        ac["energy_final"] = ac["E_total"]
        results_rows.append(ac)

        save_metadata(sub_dir, extra={"campaign": "seed_study", "seed": seed})

        print(f"  Seed={seed} | E={ac['E_total']:.4f} | A1={ac['A10']:.4f} | "
              f"A2={ac['A20']:.4f} | mode2_frac={ac['mode2_fraction']:.4f}")

    if pd is not None:
        df = pd.DataFrame(results_rows)
        df.to_csv(os.path.join(out_dir, "seed_study.csv"), index=False)
    else:
        with open(os.path.join(out_dir, "seed_study.csv"), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results_rows[0].keys())
            writer.writeheader()
            writer.writerows(results_rows)

    for key in ["A10", "A20", "A30", "mode2_fraction"]:
        vals = [r[key] for r in results_rows]
        print(f"  {key}: mean={np.mean(vals):.6f}  std={np.std(vals):.6f}")

    print(f"\n  Resultats sauvegardes dans {out_dir}/seed_study.csv")
    return results_rows


# =================================================================
# CAMPAGNE 05 : ARCHITECTURE STUDY
# =================================================================
# ═════════════════════════════════════════════════════════════════
# CAMPAGNE 09 : ARCHITECTURE STUDY (protocole figé v5)
# Seule variable : capacité du réseau (small / baseline / large).
# Re=500, E_target=0.25, base Fourier 6×5 (30 coeffs), seeds 1,2,6.
# Baseline : réutilisée depuis 04_seed_study (même config, pas de retraining).
# ================================================================
def run_architecture_study():
    print("\n" + "=" * 65)
    print("CAMPAGNE 09 : ARCHITECTURE STUDY (protocole figé) — Re=500, base 6×5, seeds 0,1,2")
    print("=" * 65)

    out_dir = os.path.join(OUT, "09_architecture")
    os.makedirs(out_dir, exist_ok=True)

    ARCH_SEEDS = [0, 1, 2]
    RE_FIX     = 500
    N_MX_FIX   = 6
    N_MT_FIX   = 5
    SEED_DIR   = os.path.join(OUT, "04_seed_study")

    results_rows = []

    def _last_losses(history_dict=None, csv_path=None):
        """Pertes de contrôle numérique (dernière époque)."""
        row = {"final_loss": None, "PDE_loss": None, "BC_loss": None,
               "diss_loss": None, "ctrl_loss": None, "var_loss": None}
        if history_dict is not None and len(history_dict.get("total", [])) > 0:
            row["final_loss"] = float(history_dict["total"][-1])
            row["PDE_loss"]   = float(history_dict["pde"][-1])
            row["BC_loss"]    = float(history_dict["bc"][-1])
            row["diss_loss"]  = float(history_dict["diss"][-1])
            row["ctrl_loss"]  = float(history_dict["ctrl"][-1])
            row["var_loss"]   = float(history_dict["var"][-1])
        elif csv_path and os.path.exists(csv_path):
            try:
                df_last = pd.read_csv(csv_path).tail(1)
                if len(df_last) > 0:
                    row["final_loss"] = float(df_last["total"].iloc[-1])
                    row["PDE_loss"]   = float(df_last["pde"].iloc[-1])
                    row["BC_loss"]    = float(df_last["bc"].iloc[-1])
                    row["diss_loss"]  = float(df_last["diss"].iloc[-1])
                    row["ctrl_loss"]  = float(df_last["ctrl"].iloc[-1])
                    row["var_loss"]   = float(df_last["var"].iloc[-1])
            except Exception:
                pass
        return row

    for arch_name, arch_cfg in ARCHITECTURES.items():
        for seed in ARCH_SEEDS:
            print(f"\n--- Architecture: {arch_name} | seed={seed} ---")
            set_seed(seed)
            sub_dir = os.path.join(out_dir, arch_name, f"seed_{seed}")
            os.makedirs(sub_dir, exist_ok=True)

            reused = False
            if arch_name == "baseline":
                src = os.path.join(SEED_DIR, f"seed_{seed}", "model.pt")
                if os.path.exists(src):
                    model = UltraPINN(arch_config=None, n_mx=N_MX_FIX, n_mt=N_MT_FIX,
                                      lx=LX, ly=LY, basis_type="fourier").to(device)
                    model.load_state_dict(torch.load(src, map_location="cpu", weights_only=False))
                    model.eval()
                    reused = True
                    hist  = None
                    print(f"  [REUSE] baseline seed={seed} depuis 04_seed_study (pas de retraining)")
                else:
                    model, hist, _ = train_ultra(
                        re_list=[RE_FIX], n_mx=N_MX_FIX, n_mt=N_MT_FIX,
                        out_dir=sub_dir,
                    )
            else:
                model, hist, _ = train_ultra(
                    re_list=[RE_FIX], n_mx=N_MX_FIX, n_mt=N_MT_FIX,
                    arch_config=arch_cfg, out_dir=sub_dir,
                )

            ac = analyze_control(model, Re=RE_FIX)
            ac["architecture"] = arch_name
            ac["seed"]            = seed
            ac["Re"]              = RE_FIX
            ac["n_mx"]            = N_MX_FIX
            ac["n_mt"]            = N_MT_FIX
            ac["n_modes"]         = N_MX_FIX * N_MT_FIX
            ac["E_target"]        = E_TARGET
            ac["A2"]              = ac["A20"]
            ac["energy_final"]    = ac["E_total"]
            ac["n_params"]        = sum(p.numel() for p in model.parameters())
            ac["reused"]          = reused
            if reused:
                ac["reused_source"] = os.path.join("04_seed_study", f"seed_{seed}")
                losses = _last_losses(csv_path=os.path.join(SEED_DIR, f"seed_{seed}", "training_history.csv"))
            else:
                ac["reused_source"] = ""
                losses = _last_losses(history_dict=hist)
            ac.update(losses)
            results_rows.append(ac)

            save_metadata(sub_dir, extra={"campaign": "architecture", "arch": arch_name,
                                          "seed": seed, "reused": reused})

            print(f"  Arch={arch_name} seed={seed} | params={ac['n_params']} | "
                  f"E={ac['E_total']:.4f} | f2={ac['mode2_fraction']:.4f} | "
                  f"ft={ac['temporal_fraction']:.4f} | A2={ac['A2']:.4f} | "
                  f"dominant=({ac['dominant_i']},{ac['dominant_j']})")

    _cols = ["architecture", "seed", "Re", "n_mx", "n_mt", "n_modes", "E_target",
             "E_total", "mode2_fraction", "fourier_mode2_fraction", "temporal_fraction",
             "A2", "dominant_i", "dominant_j", "modal_coverage", "reconstruction_rmse",
             "n_params", "reused", "reused_source",
             "final_loss", "PDE_loss", "BC_loss", "diss_loss", "ctrl_loss", "var_loss"]
    if pd is not None:
        df = pd.DataFrame(results_rows)
        df = df[[c for c in _cols if c in df.columns]]
        df.to_csv(os.path.join(out_dir, "architecture_results.csv"), index=False)
    else:
        with open(os.path.join(out_dir, "architecture_results.csv"), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=_cols)
            writer.writeheader()
            writer.writerows(results_rows)

    print(f"\n  Resultats sauvegardes dans {out_dir}/architecture_results.csv")
    return results_rows


# =================================================================
# CAMPAGNE 06 : SAMPLING STUDY
# =================================================================
def run_sampling_study():
    print("\n" + "=" * 65)
    print("CAMPAGNE 06 : SAMPLING STUDY — Re=500")
    print("=" * 65)

    out_dir = os.path.join(OUT, "06_sampling")
    os.makedirs(out_dir, exist_ok=True)

    results_rows = []

    for samp_name, (n_phys, n_bc) in SAMPLES.items():
        print(f"\n--- Sampling: {samp_name} (N_PHYS={n_phys}, N_BC={n_bc}) ---")
        set_seed(42)
        sub_dir = os.path.join(out_dir, samp_name)

        model, hist, _ = train_ultra(
            re_list=[500],
            n_phys=n_phys,
            n_bc=n_bc,
            out_dir=sub_dir,
        )

        ac = analyze_control(model, Re=500)
        ac["sampling"] = samp_name
        ac["n_phys"] = n_phys
        ac["n_bc"] = n_bc
        ac["energy_final"] = ac["E_total"]
        results_rows.append(ac)

        save_metadata(sub_dir, extra={"campaign": "sampling", "sampling": samp_name, "seed": 42})

        print(f"  Samp={samp_name} | E={ac['E_total']:.4f} | A1={ac['A10']:.4f} | "
              f"A2={ac['A20']:.4f} | mode2_frac={ac['mode2_fraction']:.4f}")

    if pd is not None:
        df = pd.DataFrame(results_rows)
        df.to_csv(os.path.join(out_dir, "sampling_results.csv"), index=False)
    else:
        with open(os.path.join(out_dir, "sampling_results.csv"), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results_rows[0].keys())
            writer.writeheader()
            writer.writerows(results_rows)

    print(f"\n  Resultats sauvegardes dans {out_dir}/sampling_results.csv")
    return results_rows


# =================================================================
# CAMPAGNE 07 : MODE COUNT STUDY
# =================================================================
def run_mode_count_study():
    print("\n" + "=" * 65)
    print("CAMPAGNE 07 : MODE COUNT STUDY — Re=500")
    print("=" * 65)

    out_dir = os.path.join(OUT, "07_mode_count")
    os.makedirs(out_dir, exist_ok=True)

    results_rows = []

    for n_mx, n_mt in MODE_CONFIGS:
        label = f"mx{n_mx}_mt{n_mt}"
        print(f"\n--- Mode count: N_MX={n_mx}, N_MT={n_mt} ---")
        set_seed(42)
        sub_dir = os.path.join(out_dir, label)

        model_path = os.path.join(sub_dir, "model.pt")
        if os.path.exists(model_path):
            print(f"  [SKIP] model.pt already exists for {label}, loading...")
            model = UltraPINN(arch_config=None, n_mx=n_mx, n_mt=n_mt,
                              lx=LX, ly=LY, basis_type="fourier").to(device)
            model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
            model.eval()
        else:
            model, hist, _ = train_ultra(
                re_list=[500],
                n_mx=n_mx, n_mt=n_mt,
                out_dir=sub_dir,
            )

        ac = analyze_control(model, Re=500)
        ac["n_mx"] = n_mx
        ac["n_mt"] = n_mt
        ac["n_modes"] = n_mx * n_mt
        ac["energy_final"] = ac["E_total"]
        results_rows.append(ac)

        save_metadata(sub_dir, extra={"campaign": "mode_count", "n_mx": n_mx, "n_mt": n_mt, "seed": 42})

        print(f"  {label} | E={ac['E_total']:.4f} | A1={ac['A10']:.4f} | "
              f"A2={ac['A20']:.4f} | mode2_frac={ac['mode2_fraction']:.4f}")

    if pd is not None:
        df = pd.DataFrame(results_rows)
        df.to_csv(os.path.join(out_dir, "mode_count_results.csv"), index=False)
    else:
        with open(os.path.join(out_dir, "mode_count_results.csv"), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results_rows[0].keys())
            writer.writeheader()
            writer.writerows(results_rows)

    print(f"\n  Resultats sauvegardes dans {out_dir}/mode_count_results.csv")
    return results_rows


# =================================================================
# CAMPAGNE 08 : TEMPORAL STUDY
# =================================================================
def run_temporal_study():
    print("\n" + "=" * 65)
    print("CAMPAGNE 08 : TEMPORAL STUDY — Re=500")
    print("=" * 65)

    out_dir = os.path.join(OUT, "08_temporal")
    os.makedirs(out_dir, exist_ok=True)

    results_rows = []

    for n_mt in TEMPORAL_CONFIGS:
        print(f"\n--- Temporal: N_MT={n_mt} ---")
        set_seed(42)
        sub_dir = os.path.join(out_dir, f"mt{n_mt}")

        model, hist, _ = train_ultra(
            re_list=[500],
            n_mx=6, n_mt=n_mt,
            out_dir=sub_dir,
        )

        ac = analyze_control(model, Re=500)
        ac["n_mt"] = n_mt

        _lx = LX
        _n_mt = n_mt
        xs = torch.linspace(0, _lx, 200, dtype=DTYPE, device=device).view(-1, 1)
        ts = torch.linspace(0, T_MAX, 200, dtype=DTYPE, device=device).view(-1, 1)
        X, T = torch.meshgrid(xs[:,0], ts[:,0], indexing='ij')
        Xf, Tf = X.reshape(-1,1), T.reshape(-1,1)
        Re_tensor = torch.full_like(Xf, 500.0)
        with torch.no_grad():
            U = model.U_lid(Xf, Tf, Re_tensor).cpu().numpy().ravel()
        U_grid = U.reshape(200, 200)
        dx = _lx / 199
        dt = T_MAX / 199
        x_arr = xs.cpu().numpy().ravel()
        t_arr = ts.cpu().numpy().ravel()

        E_j = {}
        domain_area_t = _lx * T_MAX
        for j in range(max(_n_mt, 5)):
            if j >= _n_mt:
                E_j[f"E_j={j}"] = 0.0
                continue
            e_j = 0.0
            for i in range(6):
                si = np.sin((i+1)*np.pi*x_arr/_lx)
                cj = np.cos(j*np.pi*t_arr)
                outer = np.outer(si, cj)
                norm2 = np.sum(outer**2) * dx * dt + 1e-30
                coeff = np.sum(U_grid * outer) * dx * dt / norm2
                e_j += coeff**2 * norm2
            # Meme normalisation que E_total (moyenne sur le domaine spatio-temporel)
            e_j /= domain_area_t
            E_j[f"E_j={j}"] = float(e_j)

        ac.update(E_j)
        # E_0 : energie stationnaire (j=0) -> fraction restante apres modes temporels
        E_j0 = E_j.get("E_j=0", 0.0)
        ac["E_mode0"] = E_j0
        ac["energy_final"] = ac["E_total"]
        results_rows.append(ac)

        save_metadata(sub_dir, extra={"campaign": "temporal", "n_mt": n_mt, "seed": 42})

        ej_str = " | ".join(f"j={j}:" + f"{E_j.get(f'E_j={j}', 0.):.4f}" for j in range(min(_n_mt, 4)))
        print(f"  N_MT={n_mt} | {ej_str} | temporal_frac={ac['temporal_fraction']:.4f}")

    if pd is not None:
        df = pd.DataFrame(results_rows)
        df.to_csv(os.path.join(out_dir, "temporal_results.csv"), index=False)
    else:
        with open(os.path.join(out_dir, "temporal_results.csv"), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results_rows[0].keys())
            writer.writeheader()
            writer.writerows(results_rows)

    print(f"\n  Resultats sauvegardes dans {out_dir}/temporal_results.csv")
    return results_rows


# =================================================================
# CAMPAGNE 09 : ASPECT RATIO
# =================================================================
def run_aspect_ratio():
    print("\n" + "=" * 65)
    print("CAMPAGNE 09 : ASPECT RATIO — Re=500")
    print("=" * 65)

    out_dir = os.path.join(OUT, "09_aspect_ratio")
    os.makedirs(out_dir, exist_ok=True)

    geometries = {
        "square":      (1.0, 1.0),
        "rectangular": (2.0, 1.0),
    }

    results_rows = []

    for geom_name, (lx_val, ly_val) in geometries.items():
        print(f"\n--- Geometry: {geom_name} (LX={lx_val}, LY={ly_val}) ---")
        set_seed(42)
        sub_dir = os.path.join(out_dir, geom_name)

        model, hist, _ = train_ultra(
            re_list=[500],
            lx=lx_val, ly=ly_val,
            out_dir=sub_dir,
        )

        ac = analyze_control(model, Re=500, lx=lx_val, ly=ly_val)
        ac["geometry"] = geom_name
        ac["lx"] = lx_val
        ac["ly"] = ly_val
        ac["energy_final"] = ac["E_total"]
        results_rows.append(ac)

        save_metadata(sub_dir, extra={"campaign": "aspect_ratio", "geometry": geom_name,
                                       "lx": lx_val, "ly": ly_val, "seed": 42})

        print(f"  Geom={geom_name} | E={ac['E_total']:.4f} | A1={ac['A10']:.4f} | "
              f"A2={ac['A20']:.4f} | mode2_frac={ac['mode2_fraction']:.4f}")

    if pd is not None:
        df = pd.DataFrame(results_rows)
        df.to_csv(os.path.join(out_dir, "aspect_ratio_results.csv"), index=False)
    else:
        with open(os.path.join(out_dir, "aspect_ratio_results.csv"), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results_rows[0].keys())
            writer.writeheader()
            writer.writerows(results_rows)

    print(f"\n  Resultats sauvegardes dans {out_dir}/aspect_ratio_results.csv")
    return results_rows


# =================================================================
# CAMPAGNE 10 : UNSEEN RE
# =================================================================
def run_unseen_Re():
    print("\n" + "=" * 65)
    print("CAMPAGNE 10 : INTERPOLATION INTER-RE (continuité de la loi apprise)")
    print("NOTE : cette campagne mesure la consistance d'interpolation de U_lid(x,t,Re)")
    print("       sur des Re non vus. Ce n'est PAS une validation d'optimalité à ces Re,")
    print("       qui nécessiterait une référence indépendante (ex. LBM).")
    print("=" * 65)

    out_dir = os.path.join(OUT, "10_unseen_Re")
    os.makedirs(out_dir, exist_ok=True)

    set_seed(42)
    model, hist, _ = train_ultra(
        re_list=[100, 500, 1000],
        out_dir=os.path.join(out_dir, "train"),
    )

    results_rows = []
    for unseen_Re in UNSEEN_RE:
        print(f"\n--- Unseen Re: {unseen_Re} ---")
        ac = analyze_control(model, Re=unseen_Re)
        ac["Re"] = unseen_Re
        ac["is_unseen"] = True
        results_rows.append(ac)

        print(f"  Re={unseen_Re} | E={ac['E_total']:.4f} | A1={ac['A10']:.4f} | "
              f"A2={ac['A20']:.4f} | mode2_frac={ac['mode2_fraction']:.4f}")

    for seen_Re in [100, 500, 1000]:
        ac = analyze_control(model, Re=seen_Re)
        ac["Re"] = seen_Re
        ac["is_unseen"] = False
        results_rows.append(ac)

    if pd is not None:
        df = pd.DataFrame(results_rows)
        df.to_csv(os.path.join(out_dir, "unseen_re_results.csv"), index=False)
    else:
        with open(os.path.join(out_dir, "unseen_re_results.csv"), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results_rows[0].keys())
            writer.writeheader()
            writer.writerows(results_rows)

    save_metadata(out_dir, extra={"campaign": "unseen_re_interpolation",
                                   "seen_re": [100, 500, 1000],
                                   "unseen_re": UNSEEN_RE, "seed": 42,
                                   "interpretation": "interpolation consistency across Reynolds, not independent validation"})

    print(f"\n  Resultats sauvegardes dans {out_dir}/unseen_re_results.csv")
    return model, results_rows


# =================================================================
# CAMPAGNE 11 : SYMBOLIC REGRESSION (enhanced)
# =================================================================
def run_symbolic_mode():
    print("\n" + "=" * 65)
    print("CAMPAGNE 11 : SYMBOLIC REGRESSION — representabilite")
    print("=" * 65)

    out_dir = os.path.join(OUT, "11_symbolic_LOCO")
    os.makedirs(out_dir, exist_ok=True)

    set_seed(42)
    model, hist, _ = train_ultra(
        re_list=RE_LIST,
        out_dir=os.path.join(out_dir, "train"),
    )

    all_data_flat, all_data_by_re = extract_lid(model, nx=80, nt=80)
    np.savetxt(os.path.join(out_dir, "lid_all.txt"), all_data_flat,
               header="x t Re_norm U_lid")
    for Re in RE_LIST:
        np.savetxt(os.path.join(out_dir, f"lid_Re{Re}.txt"), all_data_by_re[Re],
                   header="x t Re_norm U_lid")

    sr_model, sr_expr, sr_mse, sr_r2, sr_mae, sr_rmse, sr_rel2, sr_maxe = \
        run_pysr(all_data_flat, "global fit", niter=500)
    sr_expr_str = sr_expr

    loco_results = loco_ultra(all_data_by_re)

    sr_rows = []
    sr_rows.append({
        "Re": "global",
        "expr": sr_expr_str,
        "mse_train": sr_mse, "r2_train": sr_r2,
        "mae_train": sr_mae, "rmse_train": sr_rmse,
        "relative_L2_train": sr_rel2, "max_abs_error_train": sr_maxe,
    })
    for Re in RE_LIST:
        lr = loco_results[Re]
        sr_rows.append({
            "Re": str(Re),
            "expr": lr['expr'],
            "mse_train": lr.get('mse_train', None),
            "r2_train": lr.get('r2_train', None),
            "mse_test": lr['mse_test'], "r2_test": lr['r2_test'],
            "mae_test": lr.get('mae_test', None),
            "rmse_test": lr.get('rmse_test', None),
            "relative_L2_test": lr.get('relative_L2_test', None),
            "max_abs_error_test": lr.get('max_abs_error_test', None),
        })

    if pd is not None:
        df = pd.DataFrame(sr_rows)
        df.to_csv(os.path.join(out_dir, "symbolic_results.csv"), index=False)
    else:
        with open(os.path.join(out_dir, "symbolic_results.csv"), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sr_rows[0].keys())
            writer.writeheader()
            writer.writerows(sr_rows)

    print(f"\n{'='*65}")
    print("RESUME SYMBOLIQUE — REPRESENTABILITE")
    print(f"{'='*65}")
    print(f"  Global fit R2={sr_r2:.4f}  MAE={sr_mae:.4e}  RMSE={sr_rmse:.4e}")
    for Re in RE_LIST:
        lr = loco_results[Re]
        print(f"  LOCO Re={Re}  R2={lr['r2_test']:.4f}  MAE={lr.get('mae_test', 0.):.4e}")

    make_all_figures(model, hist, loco_results, all_data_by_re, out_dir=out_dir)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

    save_metadata(out_dir, extra={"campaign": "symbolic", "seed": 42})

    print(f"\n  Resultats sauvegardes dans {out_dir}/")
    return model, sr_rows, loco_results


# =================================================================
# CAMPAGNE 12 : CONVERGENCE / INCERTITUDE
# =================================================================
def run_convergence():
    print("\n" + "=" * 65)
    print("CAMPAGNE 12 : CONVERGENCE — incertitude")
    print("=" * 65)

    out_dir = os.path.join(OUT, "12_convergence_uncertainty")
    os.makedirs(out_dir, exist_ok=True)

    set_seed(42)
    model, hist, residual_history = train_ultra(
        re_list=RE_LIST,
        out_dir=out_dir,
        collect_residuals=True,
    )

    all_data_flat, all_data_by_re = extract_lid(model, nx=80, nt=80)
    loco_results = loco_ultra(all_data_by_re)

    make_all_figures(model, hist, loco_results, all_data_by_re, out_dir=out_dir)

    save_metadata(out_dir, extra={"campaign": "convergence", "seed": 42,
                                   "residual_stats_recorded": len(residual_history)})

    print(f"\n  Resultats sauvegardes dans {out_dir}/")
    print(f"  training_history.csv et residual_stats.csv enregistres.")
    return model, hist, loco_results


# =================================================================
# CAMPAGNE 13 : PARAMETRISATION DU CONTROLE (R2-2)
# =================================================================
def run_parametrization_study():
    """
    Compare deux familles de parametrisation du controle :
      - fourier        : sin((i+1) pi x / Lx) * cos(j pi t)  [baseline]
      - chebyshev_mod  : x(Lx-x) * T_i(2x/Lx - 1) * cos(j pi t)
                         (smooth, satisfait U(0)=U(Lx)=0)
    IMPORTANT — metrique de comparaison SCIENTIFIQUEMENT PROPRE :
      Les deux familles sont evaluees sur une grille (x,t) commune, puis
      projetees dans UNE BASE COMMUNE de Fourier par analyze_control().
      La metrique principale est donc `fourier_mode2_fraction` (= fraction
      d'energie du mode spatial 2 en projection Fourier) dans les DEUX cas.
      On conserve AUSSI les coefficients NATIFS de chaque base
      (native_c_ij) pour documenter la parametrisation elle-meme, mais ils
      ne sont PAS directement comparables entre Fourier et Chebyshev (echelles
      differentes, enveloppe x(Lx-x) de max 1/4 pour Lx=1).
    """
    print("\n" + "=" * 65)
    print("CAMPAGNE 13 : PARAMETRISATION DU CONTROLE — Re=500")
    print("=" * 65)

    out_dir = os.path.join(OUT, "13_parametrization")
    os.makedirs(out_dir, exist_ok=True)

    Re_PARAM = [100, 500, 1000]
    basis_families = ["fourier", "chebyshev_mod"]

    results_rows = []
    for btype in basis_families:
        for Re in Re_PARAM:
            print(f"\n--- Basis family: {btype} | Re={Re} ---")
            set_seed(42)
            sub_dir = os.path.join(out_dir, btype, f"re_{Re}")
            os.makedirs(sub_dir, exist_ok=True)
            model_path = os.path.join(sub_dir, "model.pt")

            if os.path.exists(model_path):
                model = UltraPINN(arch_config=None, n_mx=N_MX, n_mt=N_MT,
                                  lx=LX, ly=LY, basis_type=btype).to(device)
                model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
                model.eval()
                print(f"  [REUSE] {btype} Re={Re} (pas de retraining)")
            else:
                model, hist, _ = train_ultra(
                    re_list=[Re],
                    n_mx=N_MX, n_mt=N_MT,
                    basis_type=btype,
                    out_dir=sub_dir,
                )

            ac = analyze_control(model, Re=Re)
            ac["basis_type"] = btype
            ac["energy_final"] = ac["E_total"]
            # Metrique comparative principale : fraction du mode 2 **en projection Fourier**
            # communement calculee pour les deux familles. Ce n'est PAS le coefficient
            # natif de la base (Chebyshev modifie != harmonique sine).
            ac["fourier_mode2_fraction"] = ac["mode2_fraction"]
            ac["native_c20"] = ac.get(f"native_c_1_0", 0.0)
            results_rows.append(ac)

            save_metadata(sub_dir, extra={"campaign": "parametrization",
                                           "basis_type": btype, "seed": 42,
                                           "re_list": [Re]})

            print(f"  Basis={btype} Re={Re} | E={ac['E_total']:.4f} | "
                  f"Fourier-mode2-frac={ac['fourier_mode2_fraction']:.4f} | "
                  f"native_c20={ac['native_c20']:.4f}")

    if pd is not None:
        df = pd.DataFrame(results_rows)
        df.to_csv(os.path.join(out_dir, "parametrization_results.csv"), index=False)
    else:
        with open(os.path.join(out_dir, "parametrization_results.csv"), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results_rows[0].keys())
            writer.writeheader()
            writer.writerows(results_rows)

    print(f"\n  Resultats sauvegardes dans {out_dir}/parametrization_results.csv")
    return results_rows


# =================================================================
# RUN ALL CAMPAIGNS
# =================================================================
def run_all_campaigns():
    print("\n" + "=" * 65)
    print("LANCEMENT DE TOUTES LES CAMPAGNES")
    print("=" * 65)
    t0 = time.time()

    run_baseline()
    run_loss_ablation()
    run_energy_sweep()
    run_seed_study()
    run_architecture_study()
    run_sampling_study()
    run_mode_count_study()
    run_temporal_study()
    run_aspect_ratio()
    run_unseen_Re()
    run_symbolic_mode()
    run_convergence()
    run_parametrization_study()

    print(f"\n{'='*65}")
    print(f"TOUTES LES CAMPAGNES TERMINEES en {(time.time()-t0)/60:.1f} min")
    print(f"Resultats dans : {OUT}/")
    print(f"{'='*65}")


# ─── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    # Permet d'ecraser le MODE global via un argument CLI :
    #   python PINN_Lid_driven_reviewers.py <mode>
    # indispensable pour lancer plusieurs campagnes en parallele.
    _cli_mode = sys.argv[1] if len(sys.argv) > 1 else MODE
    if _cli_mode == "baseline":
        run_baseline()
    elif _cli_mode == "loss_ablation":
        run_loss_ablation()
    elif _cli_mode == "energy_sweep":
        run_energy_sweep()
    elif _cli_mode == "seed_study":
        run_seed_study()
    elif _cli_mode == "architecture":
        run_architecture_study()
    elif _cli_mode == "sampling":
        run_sampling_study()
    elif _cli_mode == "mode_count":
        run_mode_count_study()
    elif _cli_mode == "temporal":
        run_temporal_study()
    elif _cli_mode == "aspect_ratio":
        run_aspect_ratio()
    elif _cli_mode == "unseen_re":
        run_unseen_Re()
    elif _cli_mode == "symbolic":
        run_symbolic_mode()
    elif _cli_mode == "convergence":
        run_convergence()
    elif _cli_mode == "parametrization":
        run_parametrization_study()
    elif _cli_mode == "all":
        run_all_campaigns()
    else:
        print(f"Unknown MODE: {_cli_mode}")
