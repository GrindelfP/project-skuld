import torch
import torch.nn as nn
import numpy as np
from scipy import integrate

from skuld import (generate_data, scale_data, descale_result,
                   split_data, init_model, NeuralNumericalIntegration)
from skuld.model import set_global_device

la = 1.0
m1 = m2 = m3 = 0.3 / la
p1p1 = p2p2 = -(0.14 / la) ** 2
PP = -(0.7 / la) ** 2
BB = 75

A_VALS = [0, 0, 1, 1, 0, 0, 1, 1]
B_VALS = [0, 1, 0, 1, 0, 1, 0, 1]
M_VALS = [1, 1, 1, 1, 2, 2, 2, 2]
N_VALS = [2, 2, 2, 2, 3, 3, 3, 3]

N_INT_DIMS = 3
N_PARAMS = 4
INPUT_SIZE = N_INT_DIMS + N_PARAMS
adam_epochs = 1000
lbfgs_max_iter = 10
lbfgs_epochs = 80

def integrand_xyz(x, y, t, a, b, m, n):
    alp1 = x
    alp2 = y * (1.0 - x)
    RR = alp1 ** 2 * p1p1 + alp2 ** 2 * p2p2 - alp1 * alp2 * (PP - p1p1 - p2p2)
    DD = alp1 * (p1p1 + m1 ** 2) + alp2 * (p2p2 + m2 ** 2) + (1 - alp1 - alp2) * m3 ** 2 - RR
    z0 = t * DD + t / (1 + t) * RR
    return alp1 ** a * alp2 ** b * (1 - x) * t ** m / (1 + t) ** n * np.exp(-z0)


def numerical_reference(a, b, m, n):
    def it(t, x, y): return integrand_xyz(x, y, t, a, b, m, n)
    def iy(y, x): return integrate.quad(it, 0, BB, args=(x, y), limit=200)[0]
    def ix(x): return integrate.quad(iy, 0, 1, args=(x,), limit=200)[0]
    return integrate.quad(ix, 0, 1, limit=200)[0]


def build_dataset(n_samples_per_config: int):
    X_all, y_all = [], []
    for a, b, m, n in zip(A_VALS, B_VALS, M_VALS, N_VALS):
        def wrapper(X: torch.Tensor, _a=a, _b=b, _m=m, _n=n):
            xv, yv, tv = X[:, 0].numpy(), X[:, 1].numpy(), X[:, 2].numpy()
            vals = torch.tensor(integrand_xyz(xv, yv, tv, _a, _b, _m, _n), dtype=torch.float32)
            return vals.view(-1, 1)

        X_xyt, y_f = generate_data(
            func=wrapper, lower=[0.0, 0.0, 0.0], upper=[1.0, 1.0, BB],
            n_samples=n_samples_per_config, n_dim=3, dis_type="std",
        )
        params_col = torch.tensor([[float(a), float(b), float(m), float(n)]] * X_xyt.shape[0], dtype=torch.float32)
        X_all.append(torch.cat([X_xyt, params_col], dim=1))
        y_all.append(y_f)
    return torch.cat(X_all, dim=0), torch.cat(y_all, dim=0)


if __name__ == "__main__":
    set_global_device("cpu")

    N_SAMPLES = 10000
    HIDDEN_SIZE = 50
    LR_ADAM = 0.01

    print(f"--- Generating data ({N_SAMPLES} samples/config) ---")
    X_init, y_init = build_dataset(N_SAMPLES)
    X_sc, y_sc = scale_data(X_init, y_init, frange=(0, 1))
    x_tr, x_te, y_tr, y_te = split_data(X_sc, y_sc, test_size=0.1, shuffle=True)

    model = init_model(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)

    print(f"\n--- Phase 1: Training with Adam (LR={LR_ADAM}) ---")
    model.compile(criterion=nn.MSELoss(), optimizer_name='adam', learning_rate=LR_ADAM)
    model.fit(x_tr, y_tr, epochs=adam_epochs, verbose=True)

    print("\n--- Phase 2: Fine-tuning with L-BFGS ---")
    model.compile(criterion=nn.MSELoss(), optimizer_name='lbfgs', max_iter=lbfgs_max_iter)
    model.fit(x_tr, y_tr, epochs=lbfgs_epochs, verbose=True)  # epochs для L-BFGS — это итерации оптимизации

    print("\n" + "=" * 70)
    print(f"{'Params (a,b,m,n)':<20} | {'Scipy Ref':<12} | {'NNI Result':<12} | {'Rel. Error'}")
    print("-" * 70)

    x_min = torch.min(X_init, dim=0).values
    x_max = torch.max(X_init, dim=0).values


    def scale_theta(v, col):
        lo, hi = x_min[col].item(), x_max[col].item()
        return (v - lo) / (hi - lo) if hi != lo else 0.0


    for a, b, m, n in zip(A_VALS, B_VALS, M_VALS, N_VALS):
        theta_sc = [scale_theta(float(a), 3), scale_theta(float(b), 4),
                    scale_theta(float(m), 5), scale_theta(float(n), 6)]
        nni_sc = NeuralNumericalIntegration.integrate(
            model=model, n_dims=INPUT_SIZE, n_int_dims=N_INT_DIMS,
            theta=theta_sc, unit_cube=True)
        nni = descale_result(nni_sc, X_init, y_init, frange=(0, 1),
                             n_dim=INPUT_SIZE, n_int_dims=N_INT_DIMS)
        ref = numerical_reference(a, b, m, n)
        rel_err = abs(nni - ref) / abs(ref) if ref != 0 else abs(nni)

        print(f"({a}, {b}, {m}, {n})".ljust(20) +
              f" | {ref:<12.6e} | {nni:<12.6e} | {rel_err:<12.6e}")

    test_loss = model.test(x_te, y_te)
    print("=" * 70)
    print(f"Final Test Loss (MSE): {test_loss:.6e}")
