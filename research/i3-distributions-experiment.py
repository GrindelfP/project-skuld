import numpy as np
from skuld import *
from scipy import integrate
import matplotlib.pyplot as plt

path_f: str = "legacy/archive/test-i3.txt"
path_mm: str = "legacy/archive/test-i3-15.txt"
rounds: int = 1
number_of_epochs: int = 5000

def testOne(a, b, m, n, dist_type):
    la = 1.0
    m1 = 0.3 / la
    m2 = 0.3 / la
    m3 = 0.3 / la
    p1p1 = -(0.14 / la) ** 2
    p2p2 = -(0.14 / la) ** 2
    PP = -(0.7 / la) ** 2
    
    BB = 100
    k = -1
    
    def integrand_xyz(x, y, t):
        alp1 = x
        alp2 = y * (1.0 - x)
        RR = (alp1**2) * p1p1 + (alp2**2) * p2p2 - alp1 * alp2 * (PP - p1p1 - p2p2)
        DD = (alp1 * (p1p1 + m1**2) + alp2 * (p2p2 + m2**2) + (1.0 - alp1 - alp2) * m3**2 - RR)
        z0 = t * DD + t / (1.0 + t) * RR
        Fz0 = np.exp(k*z0)
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
    
    n_samples = 800000

    X_init, y_init = generate_data(
        dist_type,
        func=integrand_wrapper,
        lower=[0.0, 0.0, 0.0],
        upper=[1.0, 1.0, BB],
        n_samples=n_samples,
        n_dim=3
    )
    print(X_init.shape)

    X_scaled, y_scaled = scale_data(X_init=X_init, y_init=y_init, frange=(0, 1)) 
    x_train, x_test, y_train, y_test = split_data(X_scaled, y_scaled, test_size=0.1, shuffle=True)                       
                     
    model = init_model(input_size=3, hidden_size=25)                 
    model.compile_default(learning_rate=0.1)                    
    train_history = model.fit(x_train=x_train, y_train=y_train, epochs=number_of_epochs, verbose=True)

    test_loss = model.test(x_test=x_test, y_test=y_test)                      
    print(f"Test Loss: {test_loss:.10f}")
                    
    nni_scaled = NeuralNumericalIntegration.integrate(model=model, n_dims=3, unit_cube=True)                      
    nni_result = descale_result(nni_scaled, X_init=X_init, y_init=y_init, frange=(0, 1), n_dim=3)                      
    print(f"I({a}, {b}, {m}, {n}) = {nni_result}")
    print(f"S({a}, {b}, {m}, {n}) = {nni_scaled}") 
    num_result, num_error = integrate.quad(integrand_x, 0, 1, limit=200)
    print(f"N({a}, {b}, {m}, {n}) = {num_result}")
    
    abs_error: float = abs(nni_result - num_result)

    print("==========================================")
    print(f"|I_nni - I_num| = {abs_error}") 
    print("==========================================")
    
    return train_history, abs_error


if __name__ == "__main__":
    A, B, M, N = [0, 0], [0, 1], [1, 1], [2, 2]
    dist_types = ["std", "grd", "exp", "log", "mix"]
    labels = {
        "std": "Standard Uniform",
        "grd": "Uniform Grid",
        "exp": "Exponential",
        "log": "Logarithmic",
        "mix": "Mixed Density"
    }

    all_histories = {dt: [] for dt in dist_types}
    all_errors = {dt: [] for dt in dist_types}

    aa, bb, mm, nn = A[0], B[0], M[0], N[0]
    print(f"Starting comparison for I({aa}, {bb}, {mm}, {nn})...")

    for dt in dist_types:
        history, error = testOne(aa, bb, mm, nn, dt)
        all_histories[dt] = history
        all_errors[dt] = error

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, dt in enumerate(dist_types):
        axes[i].semilogy(all_histories[dt], label=labels[dt], color='blue', alpha=0.7)
        axes[i].set_title(f"Learning Curve: {labels[dt]}")
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("MAE Loss")
        axes[i].grid(True, which="both", ls="-", alpha=0.2)
        axes[i].legend()

    error_values = [all_errors[dt] for dt in dist_types]
    display_labels = [labels[dt] for dt in dist_types]
    
    bars = axes[5].bar(display_labels, error_values, color=['gray', 'blue', 'green', 'orange', 'red'])
    axes[5].set_yscale('log')
    axes[5].set_title("Integration Error Comparison (Log Scale)")
    axes[5].set_ylabel("Absolute Error |I_nni - I_num|")
    plt.setp(axes[5].get_xticklabels(), rotation=45, ha="right")
    
    for bar in bars:
        height = bar.get_height()
        axes[5].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1e}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()
