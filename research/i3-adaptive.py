import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from scipy import integrate

from skuld import *

first_stage_epochs: int = 1000
second_stage_epochs: int = 0

path_f: str = "legacy/archive/test-i3-adapt.txt"
path_mm: str = "legacy/archive/test-i3-15-adapt.txt"
rounds: int = 1
la = 1.0
m1 = 0.3 / la
m2 = 0.3 / la
m3 = 0.3 / la
p1p1 = -(0.14 / la) ** 2
p2p2 = -(0.14 / la) ** 2
PP = -(0.7 / la) ** 2
BB = 100
k = -1


def test_one(a_, b_, m_, n_):
    a, b, m, n = a_, b_, m_, n_

    def integrand_xyz(x, y, t):
        alp1 = x
        alp2 = y * (1.0 - x)
        RR = (alp1 ** 2) * p1p1 + (alp2 ** 2) * p2p2 - alp1 * alp2 * (PP - p1p1 - p2p2)
        DD = (alp1 * (p1p1 + m1 ** 2) + alp2 * (p2p2 + m2 ** 2) + (1.0 - alp1 - alp2) * m3 ** 2 - RR)
        z0 = t * DD + t / (1.0 + t) * RR
        Fz0 = np.exp(k * z0)
        jacobian = (1.0 - x)
        alp1_power = alp1 ** a
        alp2_power = alp2 ** b

        return alp1_power * alp2_power * jacobian * (t ** m) / ((1.0 + t) ** n) * Fz0

    def integrand_wrapper(X):
        return integrand_xyz(x=X[:, 0], y=X[:, 1], t=X[:, 2])

    def integrand_t(t, x, y):
        return integrand_xyz(x, y, t)

    def integrand_y(y, x):
        val, _ = integrate.quad(integrand_t, 0, BB, args=(x, y), limit=200)
        return val

    def integrand_x(x):
        val, _ = integrate.quad(integrand_y, 0, 1, args=(x,), limit=200)
        return val

    uniform_grid_samples = 50
    std_distr_size = uniform_grid_samples ** 3

    X_init, y_init = generate_data(
        func=integrand_wrapper, lower=[0.0, 0.0, 0.0],
        upper=[1.0, 1.0, BB], n_samples=std_distr_size, n_dim=3, dis_type="std")
    X_scaled, y_scaled = scale_data(X_init=X_init, y_init=y_init, frange=(0, 1))
    x_train, x_test, y_train, y_test = split_data(X_scaled, y_scaled, test_size=0.1, shuffle=True)

    model = init_model(input_size=3, hidden_size=25)

    model.compile(nn.MSELoss(), 'adamw', learning_rate=0.1, weight_decay=1e-4)
    hist1 = model.fit(x_train, y_train, epochs=first_stage_epochs, verbose=True)

    model.compile(nn.MSELoss(), 'lbfgs', learning_rate=0.01, max_iter=10)
    hist2 = model.fit(x_train, y_train, epochs=second_stage_epochs, verbose=True)

    train_history = hist1 + hist2

    plt.semilogy(train_history, 'b-', linewidth=1.5, marker='o', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('MAE Loss (log scale)')
    plt.title('Training MAE Loss Over Epochs')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # test_loss = model.test(x_test=x_test, y_test=y_test)
    # print(f"Test Loss: {test_loss:.10f}")

    nni_scaled = NeuralNumericalIntegration.integrate(model=model, unit_cube=True, n_dims=3)
    nni_result = descale_result(nni_scaled, X_init=X_init, y_init=y_init, frange=(0, 1), n_dim=3)
    print(f"I({a}, {b}, {m}, {n}) = {nni_result}")
    print(f"S({a}, {b}, {m}, {n}) = {nni_scaled}")
    num_result, num_error = integrate.quad(integrand_x, 0, 1, limit=200)
    print(f"N({a}, {b}, {m}, {n}) = {num_result}")

    print("==========================================")
    print(f"|I_nni - I_num| = {abs(nni_result - num_result)}")
    print("==========================================")

    with open(path_f, "a") as f:
        line = f"I({a}, {b}, {m}, {n}) = {nni_result:.15e}. N({a}, {b}, {m}, {n}) = {num_result:.15e}. Error |I - N| = {abs(nni_result - num_result):.6e}\n"
        f.write(line)

    return abs(nni_result - num_result)


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    A, B, M, N = [0], [0], [2], [2]
    errors_m = []
    for _ in range(rounds):
        errors_a = []
        with open(path_f, "a") as f:
            f.write(f"\n\n==== {timestamp} ====\n")
        for aa, bb, mm, Nn in zip(A, B, M, N):
            print(f"I({aa}, {bb}, {mm}, {Nn})")
            error = test_one(aa, bb, mm, Nn)
            errors_a.append(error)
        errors_m.append(errors_a)

    errors_array = np.array(errors_m)
    means = errors_array.mean(axis=0)
    medians = np.median(errors_array, axis=0)

    with open(path_mm, "a") as f:
        f.write(f"\n\n==== {timestamp} ====\nmean, median\n")
        for m, med in zip(means, medians):
            f.write(f"{m:.6e}, {med:.6e}\n")  # используем научный формат для точности

    indices = np.arange(len(means))
    combo_labels = [f"I({aa},{bb},{mm},{nn})" for aa, bb, mm, nn in zip(A, B, M, N)]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(indices, means, color='skyblue', edgecolor='navy', alpha=0.8)
    plt.yscale('log')
    plt.xlabel('Интегралы')
    plt.ylabel('Средняя абсолютная ошибка (лог. шкала)')
    plt.title('Средняя абсолютная ошибка для каждого интеграла (логарифмическая шкала)')
    plt.xticks(indices, combo_labels, rotation=45, ha='right')
    plt.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)

    for i, (bar, err) in enumerate(zip(bars, means)):
        if err > 0:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{err:.1e}', ha='center', va='bottom', fontsize=8, rotation=30)

    plt.tight_layout()
    plt.savefig(f"errors_log_{timestamp}.png", dpi=150)
    plt.show()
