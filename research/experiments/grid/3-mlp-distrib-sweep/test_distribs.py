import sys
import csv
import itertools
import datetime
import time
from pathlib import Path

import numpy as np
import torch
from scipy import integrate

from skuld import (generate_data, scale_data, descale_result,
                   split_data, init_model, NeuralNumericalIntegration)
from skuld.model import set_global_device

# ─────────────────────────────────────────────────────────────────────
# Output path
# ─────────────────────────────────────────────────────────────────────
def _unique_csv(stem: str) -> Path:
    base = Path(__file__).parent.absolute()
    today = datetime.date.today().strftime("%Y-%m-%d")
    i = 1
    while True:
        p = base / f"{stem}_{today}-{i}.csv"
        if not p.exists():
            return p
        i += 1

CSV_PATH = _unique_csv("results_t_dist")

# ─────────────────────────────────────────────────────────────────────
# Physics constants  (identical to grid_search.py)
# ─────────────────────────────────────────────────────────────────────
la   = 1.0
m1   = 0.3 / la
m2   = 0.3 / la
m3   = 0.3 / la
p1p1 = -(0.14 / la) ** 2
p2p2 = -(0.14 / la) ** 2
PP   = -(0.7  / la) ** 2
BB   = 100                   # upper integration limit for t

A_VALS = [0, 0, 1, 1, 0, 0, 1, 1]
B_VALS = [0, 1, 0, 1, 0, 1, 0, 1]
M_VALS = [1, 1, 1, 1, 2, 2, 2, 2]
N_VALS = [2, 2, 2, 2, 3, 3, 3, 3]

N_INT_DIMS = 3               # x, y, t
N_PARAMS   = 4               # a, b, m, n
INPUT_SIZE = N_INT_DIMS + N_PARAMS

# Unique (a,b,m,n) tuples used as parameter grid values
PARAM_COMBOS = list(zip(A_VALS, B_VALS, M_VALS, N_VALS))

# ─────────────────────────────────────────────────────────────────────
# Integrand
# ─────────────────────────────────────────────────────────────────────
def integrand_xyz(x, y, t, a, b, m, n):
    alp1 = x
    alp2 = y * (1.0 - x)
    RR   = alp1**2 * p1p1 + alp2**2 * p2p2 - alp1 * alp2 * (PP - p1p1 - p2p2)
    DD   = (alp1 * (p1p1 + m1**2)
            + alp2 * (p2p2 + m2**2)
            + (1 - alp1 - alp2) * m3**2 - RR)
    z0   = t * DD + t / (1 + t) * RR
    return alp1**a * alp2**b * (1 - x) * t**m / (1 + t)**n * np.exp(-z0)


# ─────────────────────────────────────────────────────────────────────
# Numerical reference  (scipy quad, computed once)
# ─────────────────────────────────────────────────────────────────────
def numerical_ref(a, b, m, n) -> float:
    def it(t, x, y): return integrand_xyz(x, y, t, a, b, m, n)
    def iy(y, x):
        v, _ = integrate.quad(it, 0, BB, args=(x, y), limit=200)
        return v
    def ix(x):
        v, _ = integrate.quad(iy, 0, 1, args=(x,), limit=200)
        return v
    result, _ = integrate.quad(ix, 0, 1, limit=200)
    return result


# ─────────────────────────────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────────────────────────────
def build_dataset(n_samples: int, t_dis_type: str,
                  decay_rate: float = 5.0,
                  split_ratio: float = 0.15,
                  density_ratio: float = 0.85,
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    # ── step 1: [x, y, t] ──────────────────────────────────────────
    def _dummy(X: torch.Tensor) -> torch.Tensor:
        """Placeholder: evaluate integrand later after param columns are added."""
        return torch.zeros(X.shape[0], 1)

    # Use generate_data just for its samplers; we'll evaluate f ourselves
    x_col = (torch.rand(n_samples) * 1.0).unsqueeze(1)          # [0,1] uniform
    y_col = (torch.rand(n_samples) * 1.0).unsqueeze(1)          # [0,1] uniform

    if t_dis_type == "std":
        t_col = (torch.rand(n_samples) * BB).unsqueeze(1)
    elif t_dis_type == "exp":
        # exponential bias near 0
        R       = float(BB)
        max_val = 1.0 - np.exp(-decay_rate * R)
        u       = torch.rand(n_samples)
        t_col   = (-(1.0 / decay_rate) * torch.log(1.0 - u * max_val)).unsqueeze(1)
    elif t_dis_type == "log":
        epsilon = 1e-3          # avoid log(0); t≥0 but peak is near 0
        t_col   = torch.logspace(np.log10(epsilon), np.log10(BB), n_samples).unsqueeze(1)
    elif t_dis_type == "mix":
        n_dense  = int(n_samples * density_ratio)
        n_sparse = n_samples - n_dense
        split_t  = BB * split_ratio
        t_dense  = torch.linspace(0.0, split_t, n_dense)
        t_sparse = torch.linspace(split_t, BB, n_sparse + 1)[1:]
        t_col    = torch.cat([t_dense, t_sparse]).unsqueeze(1)
    elif t_dis_type == "grd":
        t_col = torch.linspace(0.0, BB, n_samples).unsqueeze(1)
    else:
        raise ValueError(f"Unknown t_dis_type: '{t_dis_type}'")

    n_combos = len(PARAM_COMBOS)  # 8

    # Repeat [x,y,t] block n_combos times, tile param combos
    x_rep = x_col.repeat(n_combos, 1)
    y_rep = y_col.repeat(n_combos, 1)
    t_rep = t_col.repeat(n_combos, 1)

    params_block = torch.tensor(
        [[float(a), float(b), float(m_), float(n_)]
         for (a, b, m_, n_) in PARAM_COMBOS
         for _ in range(n_samples)],
        dtype=torch.float32,
    )

    X = torch.cat([x_rep, y_rep, t_rep, params_block], dim=1)

    # ── evaluate integrand ─────────────────────────────────────────
    xv = X[:, 0].numpy()
    yv = X[:, 1].numpy()
    tv = X[:, 2].numpy()
    av = X[:, 3].numpy().astype(int)
    bv = X[:, 4].numpy().astype(int)
    mv = X[:, 5].numpy().astype(int)
    nv = X[:, 6].numpy().astype(int)

    y_vals = torch.tensor(
        integrand_xyz(xv, yv, tv, av, bv, mv, nv), dtype=torch.float32
    ).unsqueeze(1)

    return X, y_vals


# ─────────────────────────────────────────────────────────────────────
# Single experiment
# ─────────────────────────────────────────────────────────────────────
def run_experiment(hidden_size: int, epochs: int, n_samples: int,
                   lr: float, t_dis_type: str,
                   num_refs: dict) -> dict:
    set_global_device("cpu")

    X_init, y_init = build_dataset(n_samples, t_dis_type)
    X_sc, y_sc     = scale_data(X_init, y_init, frange=(0, 1))
    x_tr, x_te, y_tr, y_te = split_data(X_sc, y_sc, test_size=0.1, shuffle=True)

    model = init_model(input_size=INPUT_SIZE, hidden_size=hidden_size)
    model.compile_default(learning_rate=lr)

    t0 = time.time()
    model.fit(x_tr, y_tr, epochs=epochs, verbose=False)
    train_time = time.time() - t0

    test_loss = model.test(x_te, y_te)

    # ── descaling helpers ──────────────────────────────────────────
    x_min = torch.min(X_init, dim=0).values
    x_max = torch.max(X_init, dim=0).values

    def scale_theta(v: float, col: int) -> float:
        lo, hi = x_min[col].item(), x_max[col].item()
        return (v - lo) / (hi - lo) if hi != lo else 0.0

    # ── integrate for every (a,b,m,n) combo ───────────────────────
    errors = []
    for a, b, m_, n_ in PARAM_COMBOS:
        theta_sc = [
            scale_theta(float(a),  3),
            scale_theta(float(b),  4),
            scale_theta(float(m_), 5),
            scale_theta(float(n_), 6),
        ]
        nni_sc = NeuralNumericalIntegration.integrate(
            model=model, n_dims=INPUT_SIZE, n_int_dims=N_INT_DIMS,
            theta=theta_sc, unit_cube=True,
        )
        nni = descale_result(
            nni_sc, X_init, y_init, frange=(0, 1),
            n_dim=INPUT_SIZE, n_int_dims=N_INT_DIMS,
        )
        ref = num_refs[(a, b, m_, n_)]
        errors.append(abs(nni - ref))

    return {
        "mean_err":  float(np.mean(errors)),
        "max_err":   float(np.max(errors)),
        "test_loss": float(test_loss),
        "train_sec": round(train_time, 1),
        "errors":    errors,
    }


# ─────────────────────────────────────────────────────────────────────
# Experiment plan
# ─────────────────────────────────────────────────────────────────────

TOP5_CONFIGS = [
    {"hidden_size": 100, "epochs": 5000, "n_samples": 27000, "lr": 0.01},
    {"hidden_size": 50, "epochs": 5000, "n_samples":  8000, "lr": 0.01},
    {"hidden_size":  150, "epochs": 5000, "n_samples": 8000, "lr": 0.01},
    {"hidden_size": 100, "epochs": 5000, "n_samples":  8000, "lr": 0.01},
    {"hidden_size": 150, "epochs": 3000, "n_samples": 27000, "lr": 0.01},
]

# 5 sampling strategies for t
T_STRATEGIES = ["std", "exp", "log", "mix", "grd"]

# CSV columns
CSV_FIELDS = [
    "timestamp", "hidden_size", "epochs", "n_samples", "lr", "t_strategy",
    "mean_err", "max_err", "test_loss", "train_sec",
    *[f"err_{a}{b}{m_}{n_}" for a, b, m_, n_ in PARAM_COMBOS],
]


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    dry_run = "--dry-run" in sys.argv
    total   = len(TOP5_CONFIGS) * len(T_STRATEGIES)
    print(f"Total experiments: {total}  (5 configs × 5 t-strategies)")
    if dry_run:
        for i, (cfg, strat) in enumerate(
                itertools.product(TOP5_CONFIGS, T_STRATEGIES), 1):
            print(f"  [{i:2d}] {cfg}  t_strategy={strat}")
        return

    # Pre-compute scipy references once
    print("Computing numerical references via scipy ... ", end="", flush=True)
    num_refs = {
        (a, b, m_, n_): numerical_ref(a, b, m_, n_)
        for a, b, m_, n_ in PARAM_COMBOS
    }
    print("done.")
    for k, v in num_refs.items():
        print(f"  ref{k} = {v:.8e}")
    print()

    write_header = not CSV_PATH.exists()
    print(f"Saving results to: {CSV_PATH.resolve()}\n")

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        run_idx = 0
        for cfg in TOP5_CONFIGS:
            for t_strat in T_STRATEGIES:
                run_idx += 1
                label = (f"hs={cfg['hidden_size']}  ep={cfg['epochs']}  "
                         f"ns={cfg['n_samples']}  lr={cfg['lr']}  "
                         f"t={t_strat}")
                print(f"[{run_idx:2d}/{total}]  {label} ...", end="", flush=True)

                result = run_experiment(**cfg, t_dis_type=t_strat,
                                        num_refs=num_refs)

                row = {
                    "timestamp":   datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    **cfg,
                    "t_strategy":  t_strat,
                    "mean_err":    f"{result['mean_err']:.6e}",
                    "max_err":     f"{result['max_err']:.6e}",
                    "test_loss":   f"{result['test_loss']:.6e}",
                    "train_sec":   result["train_sec"],
                    **{f"err_{a}{b}{m_}{n_}": f"{e:.6e}"
                       for (a, b, m_, n_), e in zip(PARAM_COMBOS, result["errors"])},
                }
                writer.writerow(row)
                f.flush()

                print(f"  mean={result['mean_err']:.3e}  "
                      f"max={result['max_err']:.3e}  "
                      f"t={result['train_sec']}s")

    print("\nAll done.")


if __name__ == "__main__":
    main()
