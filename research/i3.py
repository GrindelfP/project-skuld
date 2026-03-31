from matplotlib import pyplot as plt
from scipy import integrate
from torch import Tensor
import numpy as np
from skuld import *
import datetime

path_f: str = "legacy/archive/test-i3.txt"
path_mm: str = "test-i3-15.txt"
rounds: int = 1

def testOne(a, b, m, n):
    la = 1.0
    m1 = 0.3 / la
    m2 = 0.3 / la
    m3 = 0.3 / la
    p1p1 = -(0.14 / la) ** 2
    p2p2 = -(0.14 / la) ** 2
    PP = -(0.7 / la) ** 2
    
    BB = 100

    def integrand_xyz(x, y, t):
        alp1 = x
        alp2 = y * (1.0 - x)
        RR = (alp1**2) * p1p1 + (alp2**2) * p2p2 - alp1 * alp2 * (PP - p1p1 - p2p2)
        DD = (alp1 * (p1p1 + m1**2) + alp2 * (p2p2 + m2**2) + (1.0 - alp1 - alp2) * m3**2 - RR)
        z0 = t * DD + t / (1.0 + t) * RR
        Fz0 = np.exp(-z0)
        jacobian = (1.0 - x)
        alp1_power = alp1 ** a
        alp2_power = alp2 ** b

        return alp1_power * alp2_power * jacobian * (t ** m) / ((1.0 + t) ** n) * Fz0
    
    
    def integrand_wrapper(X: Tensor, *args) -> Tensor:
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
    std_distribution_size = uniform_grid_samples ** 3
                    
    X_init, y_init = generate_data(
        func=integrand_wrapper,
        lower=[0.0, 0.0, 0.0],
        upper=[1.0, 1.0, BB],
        n_samples=std_distribution_size,
        n_dim=3,
        dis_type="std"
    )
    X_scaled, y_scaled = scale_data(X_init=X_init, y_init=y_init, frange=(0, 1)) 
    x_train, x_test, y_train, y_test = split_data(X_scaled, y_scaled, test_size=0.1, shuffle=True)                       
                    
    number_of_epochs: int = 1000
    model = init_model(input_size=3, hidden_size=25)                 
    model.compile_default(learning_rate=0.1)                    
    model.fit(x_train=x_train, y_train=y_train, epochs=number_of_epochs, verbose=False)

    test_loss = model.test(x_test=x_test, y_test=y_test)                      
    print(f"Test Loss: {test_loss:.10f}")
                    
    nni_scaled = NeuralNumericalIntegration.integrate(model=model, n_dims=3, unit_cube=True, alphas=[], betas=[])
    nni_result = descale_result(nni_scaled, X_init=X_init, y_init=y_init, frange=(0, 1), n_dim=3)                      
    print(f"I({a}, {b}, {m}, {n}) = {nni_result}")
    print(f"S({a}, {b}, {m}, {n}) = {nni_scaled}") 
    num_result, num_error = integrate.quad(integrand_x, 0, 1, limit=200)
    print(f"N({a}, {b}, {m}, {n}) = {num_result}")
    
    print("==========================================")
    print(f"|I_nni - I_num| = {abs(nni_result - num_result)}") 
    print("==========================================")
    
    with open(path_f, "a") as file_:
        line = (f"I({a}, {b}, {m}, {n}) = "
                f"{nni_result:.15e}. N({a}, {b}, {m}, {n}) = "
                f"{num_result:.15e}. Error |I - N| = {abs(nni_result - num_result):.6e}\n")
        file_.write(line)
    
    return abs(nni_result - num_result)


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    A, B, M, N = [0, 0, 1, 1, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1], [1, 1, 1, 1, 2, 2, 2, 2], [2, 2, 2, 2, 3, 3, 3, 3]
    errors_m = []
    for _ in range(rounds):
        errors_a = []
        with open(path_f, "a") as f:
            f.write(f"\n\n==== {timestamp} ====\n")
        for aa, bb, mm, nn in zip(A, B, M, N):
            print(f"I({aa}, {bb}, {mm}, {nn})")
            error = testOne(aa, bb, mm, nn)
            errors_a.append(error)
        errors_m.append(errors_a)

    errors_array = np.array(errors_m)
    means = errors_array.mean(axis=0)
    medians = np.median(errors_array, axis=0)

    labels = [f"I[{a}, {b}, {m}, {n}]" for a, b, m, n in zip(A, B, M, N)]
    x_ = np.arange(len(labels))  # The label locations
    width = 0.35  # The width of the bars
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x_ - width / 2, means, width, label='Mean', color='skyblue', edgecolor='navy')
    ax.bar(x_ + width / 2, medians, width, label='Median', color='salmon', edgecolor='darkred')
    ax.set_ylabel('Error Value')
    ax.set_title(f'Mean vs Median Errors ({timestamp})')
    ax.set_xticks(x_)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    with open(path_mm, "a") as file:
        file.write(f"\n\n==== {timestamp} ====\nmean, median\n")
        for m_, med in zip(means, medians):
            file.write(f"{m_:.6f}, {med:.6f}\n")
                