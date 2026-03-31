from scipy import integrate
from torch import Tensor
import torch
import numpy as np
from skuld import (generate_data, scale_data, descale_result,
                   split_data, init_model, NeuralNumericalIntegration, set_global_device)

import json
from utils.paths import get_mirror_path

# ── physical constants ────────────────────────────────────────────────
la    = 1.0
m1    = 0.3 / la;  m2 = 0.3 / la;  m3 = 0.3 / la
p1p1  = -(0.14 / la) ** 2
p2p2  = -(0.14 / la) ** 2
PP    = -(0.7  / la) ** 2
BB    = 100
N_INT_DIMS = 3                       # x, y, t
N_PARAMS   = 4                       # a, b, m, n
INPUT_SIZE = N_INT_DIMS + N_PARAMS   # 7
HIDDEN_SIZE = 100

# ── parameter grid for training ───────────────────────────────────────
A_vals = [0, 0, 1, 1, 0, 0, 1, 1]
B_vals = [0, 1, 0, 1, 0, 1, 0, 1]
M_vals = [1, 1, 1, 1, 2, 2, 2, 2]
N_vals = [2, 2, 2, 2, 3, 3, 3, 3]


def integrand_xyz(x, y, t, a, b, m, n):
    alp1 = x
    alp2 = y * (1.0 - x)
    RR   = (alp1**2)*p1p1 + (alp2**2)*p2p2 - alp1*alp2*(PP - p1p1 - p2p2)
    DD   = (alp1*(p1p1 + m1**2) + alp2*(p2p2 + m2**2)
            + (1.0 - alp1 - alp2)*m3**2 - RR)
    z0   = t*DD + t/(1.0 + t)*RR
    return (alp1**a) * (alp2**b) * (1.0 - x) * (t**m) / ((1.0 + t)**n) * np.exp(-z0)


def make_wrapper(a, b, m, n):
    """Returns a wrapper that appends fixed (a,b,m,n) as extra columns."""
    def wrapper(X: Tensor) -> Tensor:
        xv = X[:, 0].numpy(); yv = X[:, 1].numpy(); tv = X[:, 2].numpy()
        vals = torch.tensor(
            integrand_xyz(xv, yv, tv, a, b, m, n), dtype=torch.float32
        )
        return vals.view(-1, 1)
    return wrapper


# ── build joint training set ──────────────────────────────────────────
def build_joint_dataset(n_samples_per_config: int = 50**3):
    X_all, y_all = [], []
    for a, b, m, n in zip(A_vals, B_vals, M_vals, N_vals):
        wrapper = make_wrapper(a, b, m, n)

        # generate (x, y, t) samples for this config
        X_xyt, y_f = generate_data(
            func=lambda X: wrapper(X),
            lower=[0.0, 0.0, 0.0],
            upper=[1.0, 1.0, BB],
            n_samples=n_samples_per_config,
            n_dim=3,
            dis_type="std",
        )

        # append parameter columns as constant tiles
        params_col = torch.tensor(
            [[float(a), float(b), float(m), float(n)]] * X_xyt.shape[0],
            dtype=torch.float32,
        )
        X_full = torch.cat([X_xyt, params_col], dim=1)   # (N, 7)
        X_all.append(X_full)
        y_all.append(y_f)

    X_joint = torch.cat(X_all, dim=0)
    y_joint = torch.cat(y_all, dim=0)
    return X_joint, y_joint


# ── train once ────────────────────────────────────────────────────────
X_init, y_init = build_joint_dataset()
X_scaled, y_scaled = scale_data(X_init, y_init, frange=(0, 1))
x_tr, x_te, y_tr, y_te = split_data(X_scaled, y_scaled,
                                     test_size=0.1, shuffle=True)

model = init_model(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
model.compile_default(learning_rate=0.1)
model.fit(x_tr, y_tr, epochs=1000, verbose=True)
print("Test loss:", model.test(x_te, y_te))


# ── evaluate for a specific (a, b, m, n) ─────────────────────────────
def nni_for(a, b, m, n):
    x_min = torch.min(X_init, dim=0).values
    x_max = torch.max(X_init, dim=0).values

    def scale_theta(raw_val, col_idx):
        lo = x_min[col_idx].item()
        hi = x_max[col_idx].item()
        return (raw_val - lo) / (hi - lo) if hi != lo else 0.0

    theta_scaled = [
        scale_theta(float(a), 3),
        scale_theta(float(b), 4),
        scale_theta(float(m), 5),
        scale_theta(float(n), 6),
    ]

    nni_scaled = NeuralNumericalIntegration.integrate(
        model=model,
        n_dims=INPUT_SIZE,
        n_int_dims=N_INT_DIMS,
        theta=theta_scaled,
        unit_cube=True,
        alphas=[],
        betas=[]
    )

    return descale_result(
        nni_scaled,
        X_init=X_init,
        y_init=y_init,
        frange=(0, 1),
        n_dim=INPUT_SIZE,
        n_int_dims=N_INT_DIMS,
    )


def numerical_for(a, b, m, n):
    def integrand_t(t, x, y): return integrand_xyz(x, y, t, a, b, m, n)
    def integrand_y(y, x):
        val, _ = integrate.quad(integrand_t, 0, BB, args=(x, y), limit=200)
        return val
    def integrand_x(x):
        val, _ = integrate.quad(integrand_y, 0, 1, args=(x,), limit=200)
        return val
    result, _ = integrate.quad(integrand_x, 0, 1, limit=200)
    return result


# ── evaluate and save results ─────────────────────────────────────────
results_data = []

for a, b, m, n in zip(A_vals, B_vals, M_vals, N_vals):
    nni = float(nni_for(a, b, m, n))
    num = float(numerical_for(a, b, m, n))
    err = abs(nni - num)

    print(f"I({a},{b},{m},{n})  NNI={nni:.8e}  NUM={num:.8e}  err={err:.3e}")

    results_data.append({
        "config": f"({a},{b},{m},{n})",
        "a": a, "b": b, "m": m, "n": n,
        "nni": nni,
        "num": num,
        "error": err
    })

# ── Save to reflected path ────────────────────────────────────────────
res_dir = get_mirror_path(__file__, "results")
res_file = res_dir / "integration_results.json"

with open(res_file, "w", encoding="utf-8") as f:
    json.dump(results_data, f, indent=4)

print(f"\n[OK] Results saved to: {res_file}")
