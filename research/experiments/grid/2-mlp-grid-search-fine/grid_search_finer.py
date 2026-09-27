import sys
import csv
import itertools
import datetime
import time

import torch
import numpy as np
from scipy import integrate

from skuld import (generate_data, scale_data, descale_result,
                   split_data, init_model, NeuralNumericalIntegration)
from skuld.model import set_global_device

from utils.paths import get_mirror_path

RES_DIR = get_mirror_path(__file__, "results")
CSV_PATH = RES_DIR / "results_grid.csv"

# ── physics ───────────────────────────────────────────────────────────
la   = 1.0
m1   = 0.3/la; m2 = 0.3/la; m3 = 0.3/la
p1p1 = -(0.14/la)**2; p2p2 = -(0.14/la)**2; PP = -(0.7/la)**2
BB   = 100

A_VALS = [0, 0, 1, 1, 0, 0, 1, 1]
B_VALS = [0, 1, 0, 1, 0, 1, 0, 1]
M_VALS = [1, 1, 1, 1, 2, 2, 2, 2]
N_VALS = [2, 2, 2, 2, 3, 3, 3, 3]

N_INT_DIMS = 3
N_PARAMS   = 4
INPUT_SIZE = N_INT_DIMS + N_PARAMS


def integrand_xyz(x, y, t, a, b, m, n):
    alp1 = x
    alp2 = y * (1.0 - x)
    RR   = alp1**2*p1p1 + alp2**2*p2p2 - alp1*alp2*(PP - p1p1 - p2p2)
    DD   = alp1*(p1p1+m1**2) + alp2*(p2p2+m2**2) + (1-alp1-alp2)*m3**2 - RR
    z0   = t*DD + t/(1+t)*RR
    return alp1**a * alp2**b * (1-x) * t**m / (1+t)**n * np.exp(-z0)


def numerical_for(a, b, m, n):
    def it(t, x, y): return integrand_xyz(x, y, t, a, b, m, n)
    def iy(y, x):
        v, _ = integrate.quad(it, 0, BB, args=(x, y), limit=200)
        return v
    def ix(x):
        v, _ = integrate.quad(iy, 0, 1, args=(x,), limit=200)
        return v
    result, _ = integrate.quad(ix, 0, 1, limit=200)
    return result


def build_dataset(n_samples_per_config: int):
    X_all, y_all = [], []
    for a, b, m, n in zip(A_VALS, B_VALS, M_VALS, N_VALS):
        def wrapper(X: torch.Tensor, _a=a, _b=b, _m=m, _n=n):
            xv = X[:, 0].numpy(); yv = X[:, 1].numpy(); tv = X[:, 2].numpy()
            vals = torch.tensor(
                integrand_xyz(xv, yv, tv, _a, _b, _m, _n), dtype=torch.float32)
            return vals.view(-1, 1)

        X_xyt, y_f = generate_data(
            func=wrapper,
            lower=[0.0, 0.0, 0.0], upper=[1.0, 1.0, BB],
            n_samples=n_samples_per_config, n_dim=3, dis_type="std",
        )
        params_col = torch.tensor(
            [[float(a), float(b), float(m), float(n)]] * X_xyt.shape[0],
            dtype=torch.float32)
        X_all.append(torch.cat([X_xyt, params_col], dim=1))
        y_all.append(y_f)

    return torch.cat(X_all, dim=0), torch.cat(y_all, dim=0)


def run_experiment(hidden_size, epochs, n_samples, lr, num_refs):
    set_global_device("cpu")

    X_init, y_init = build_dataset(n_samples)
    X_sc, y_sc = scale_data(X_init, y_init, frange=(0, 1))
    x_tr, x_te, y_tr, y_te = split_data(X_sc, y_sc, test_size=0.1, shuffle=True)

    model = init_model(input_size=INPUT_SIZE, hidden_size=hidden_size)
    model.compile_default(learning_rate=lr)

    t0 = time.time()
    model.fit(x_tr, y_tr, epochs=epochs, verbose=False)
    train_time = time.time() - t0

    test_loss = model.test(x_te, y_te)

    # ── per-integral errors ──────────────────────────────────────────
    x_min = torch.min(X_init, dim=0).values
    x_max = torch.max(X_init, dim=0).values

    def scale_theta(v, col):
        lo, hi = x_min[col].item(), x_max[col].item()
        return (v - lo) / (hi - lo) if hi != lo else 0.0

    errors = []
    for a, b, m, n in zip(A_VALS, B_VALS, M_VALS, N_VALS):
        theta_sc = [scale_theta(float(a), 3), scale_theta(float(b), 4),
                    scale_theta(float(m), 5), scale_theta(float(n), 6)]

        nni_sc = NeuralNumericalIntegration.integrate(
            model=model, n_dims=INPUT_SIZE, n_int_dims=N_INT_DIMS,
            theta=theta_sc, unit_cube=True)

        nni = descale_result(nni_sc, X_init, y_init,
                             frange=(0, 1), n_dim=INPUT_SIZE,
                             n_int_dims=N_INT_DIMS)
        errors.append(abs(nni - num_refs[(a, b, m, n)]))

    return {
        "mean_err":  float(np.mean(errors)),
        "max_err":   float(np.max(errors)),
        "test_loss": float(test_loss),
        "train_sec": round(train_time, 1),
        "errors":    errors,
    }


# ── grid definition ───────────────────────────────────────────────────
GRID = {
    "hidden_size": [125, 150, 175, 200],
    "epochs":      [4000, 5000, 6000],
    "n_samples":   [27000, 75000],
    "lr":          [0.01],
}

CSV_FIELDS = [
    "timestamp", "hidden_size", "epochs", "n_samples", "lr",
    "mean_err", "max_err", "test_loss", "train_sec",
    *[f"err_{a}{b}{m}{n}" for a, b, m, n in
      zip(A_VALS, B_VALS, M_VALS, N_VALS)],
]


def main():
    dry_run = "--dry-run" in sys.argv
    configs = list(itertools.product(*GRID.values()))
    print(f"Total configs: {len(configs)}")
    if dry_run:
        return

    # precompute numerical references once — they don't depend on hyperparams
    print("Computing numerical references... ", end="", flush=True)
    num_refs = {(a, b, m, n): numerical_for(a, b, m, n)
                for a, b, m, n in zip(A_VALS, B_VALS, M_VALS, N_VALS)}
    print("done.")

    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        keys = list(GRID.keys())
        for i, vals in enumerate(configs, 1):
            cfg = dict(zip(keys, vals))
            print(f"[{i}/{len(configs)}] {cfg} ... ", end="", flush=True)

            result = run_experiment(**cfg, num_refs=num_refs)

            row = {
                "timestamp":   datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                **cfg,
                "mean_err":    f"{result['mean_err']:.6e}",
                "max_err":     f"{result['max_err']:.6e}",
                "test_loss":   f"{result['test_loss']:.6e}",
                "train_sec":   result["train_sec"],
                **{f"err_{a}{b}{m}{n}": f"{e:.6e}"
                   for (a, b, m, n), e in zip(
                       zip(A_VALS, B_VALS, M_VALS, N_VALS), result["errors"])},
            }
            writer.writerow(row)
            f.flush()   # write immediately — safe to Ctrl+C mid-run

            print(f"mean_err={result['mean_err']:.3e}  "
                  f"max_err={result['max_err']:.3e}  "
                  f"t={result['train_sec']}s")


if __name__ == "__main__":
    main()
