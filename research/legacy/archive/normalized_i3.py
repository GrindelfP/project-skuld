import numpy as np
import torch
from scipy import integrate
from skuld import *
import datetime

# Настройки путей
path_f = "rect_results.txt"


def test_rect_integral(aa, bb, mm, nn):
    a, b, m, n = aa, bb, mm, nn
    la = 1.0
    # Параметры (m^2 > 0, P^2 < 0 для Евклидовой области)
    m1s, m2s, m3s = (0.3 / la) ** 2, (0.3 / la) ** 2, (0.3 / la) ** 2
    p1p1, p2p2, PP = -(0.14 / la) ** 2, -(0.14 / la) ** 2, -(0.7 / la) ** 2

    def integrand_rect(x, y, u):
        # Предосторожность 1: Крайние значения u
        if u >= 1.0: return 0.0
        if u <= 0.0 and m <= 0: return 0.0  # защита если m=0

        # Обратное отображение переменных
        alp1 = x
        alp2 = y * (1.0 - x)
        alp3 = 1.0 - alp1 - alp2

        # Предосторожность 2: вычисление t
        t = u / (1.0 - u)

        # Формулы из условия
        RR2 = (alp1 ** 2) * p2p2 + (alp2 ** 2) * p1p1 - (alp1 * alp2) * (PP - p1p1 - p2p2)
        DD = (alp1 * alp2) * PP + p1p1 * alp2 * alp3 + p2p2 * alp1 * alp3 + \
             alp1 * m1s + alp2 * m2s + alp3 * m3s

        z0 = t * DD + (t / (1.0 + t)) * RR2

        # Предосторожность 3: Защита от overflow в exp
        # Если z0 > 100, exp(-z0) пренебрежимо мал (~0)
        # Если z0 < -100, интеграл расходится, ставим заглушку для стабильности NNI
        if z0 > 100:
            Fz0 = 0.0
        elif z0 < -50:
            Fz0 = 1e10  # Сигнал о расходимости, но не inf
        else:
            Fz0 = np.exp(-z0)

        # Сборка функции с Якобианами
        # u^m * (1-u)^(n-m-2)
        val = (x ** a) * ((1.0 - x) ** (b + 1)) * (y ** b) * \
              (u ** m) * ((1.0 - u) ** (n - m - 2)) * Fz0

        return val

    def wrapper(X):
        # Векторная обертка для Skuld
        res = []
        for i in range(X.shape[0]):
            res.append(integrand_rect(X[i, 0].item(), X[i, 1].item(), X[i, 2].item()))
        return torch.tensor(res, dtype=torch.float32)

    # 1. Генерация данных на кубе [0, 1]^3
    X_init, y_init = generate_data(
        func=wrapper,
        lower=[0.0, 0.0, 0.0],
        upper=[1.0, 1.0, 1.0],
        n_samples=8000,  # 20^3 точек
        n_dim=3,
        dis_type="std"
    )

    # 2. Обучение
    X_scaled, y_scaled = scale_data(X_init, y_init, frange=(0, 1))
    x_train, x_test, y_train, y_test = split_data(X_scaled, y_scaled, test_size=0.1, shuffle=True)

    model = init_model(input_size=3, hidden_size=64)
    model.compile_default(learning_rate=0.01)
    model.fit(x_train, y_train, epochs=2000, verbose=False)

    # 3. Результаты
    nni_val = NeuralNumericalIntegration.integrate(model=model, n_dims=3, unit_cube=True)
    nni_result = descale_result(nni_val, X_init, y_init, frange=(0, 1), n_dim=3)

    num_result, _ = integrate.tplquad(
        lambda u, y, x: integrand_rect(x, y, u),
        0, 1, lambda x: 0, lambda x: 1, lambda x, y: 0, lambda x, y: 1
    )

    print(f"I({a},{b},{m},{n}) -> NNI: {nni_result:.6e} | Scipy: {num_result:.6e}")
    return nni_result


if __name__ == "__main__":
    # Пример для параметров, которые с большей вероятностью сойдутся
    # Если вы всё еще видите inf, увеличьте массы (m1, m2, m3)
    # или уменьшите внешние импульсы (PP).
    test_rect_integral(0, 0, 1, 3)