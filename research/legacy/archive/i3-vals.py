import numpy as np
from scipy.interpolate import griddata
from skuld import *
from scipy import integrate
import datetime

path_f: str = "vals-i3.txt"
rounds: int = 1


def testOne(aa, bb, mm, nn):
    a, b, m, n = aa, bb, mm, nn
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
    
    
    def integrand_cos(x, y, t):
            return np.cos(x) * np.cos(y) * np.cos(t)
    
    
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
    std_ditr_size = uniform_grid_samples ** 3
                    
    X_init, y_init = generate_data_uniform(
        func=integrand_wrapper,lower=[0.0, 0.0, 0.0],upper=[1.0, 1.0, BB],n_samples=std_ditr_size,n_dim=3)                       
    X_scaled, y_scaled = scale_data(X_init=X_init, y_init=y_init, frange=(0, 1)) 
    x_train, x_test, y_train, y_test = split_data(X_scaled, y_scaled, test_size=0.1, shuffle=True)                       
                    
    number_of_epochs: int = 10000 
    model = init_model(input_size=3, hidden_size=25)                 
    model.compile_default(learning_rate=0.1)                    
    train_history = model.fit(x_train=x_train, y_train=y_train, epochs=number_of_epochs, verbose=False)             

    test_loss = model.test(x_test=x_test, y_test=y_test)                      
    print(f"Test Loss: {test_loss:.10f}")
                    
    nni_scaled = NeuralNumericalIntegration.integrate(model=model, alphas=[0.0, 0.0, 0.0],betas=[1.0, 1.0, 1.0], n_dims=3)                      
    nni_result = descale_result(nni_scaled, X_init=X_init, y_init=y_init, frange=(0, 1), n_dim=3)                      
    print(f"I({a}, {b}, {m}, {n}) = {nni_result}")
    print(f"S({a}, {b}, {m}, {n}) = {nni_scaled}") 
    num_result, num_error = integrate.quad(integrand_x, 0, 1, limit=200)
    print(f"N({a}, {b}, {m}, {n}) = {num_result}")
    
    print("==========================================")
    print(f"|I_nni - I_num| = {abs(nni_result - num_result)}") 
    print("==========================================")
    
    with open(path_f, "a") as f: 
        line = f"I({a}, {b}, {m}, {n}) = {nni_result}. N({a}, {b}, {m}, {n}) = {num_result}. Error |I - N| = {abs(nni_result - num_result):.6e}\n"
        f.write(line)
    
    return abs(nni_result - num_result)


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    A, B, M, N = [0, 1, 0, 1, 1, 0, 1, 2, 0], [0, 0, 1, 0, 1, 0, 2, 0, 0], [2, 3, 3, 1, 1, 2, 5, 5, 5], [2, 3, 3, 2, 2, 3, 5, 5, 4]
    with open(path_f, "a") as f: 
        f.write(f"\n\n==== {timestamp} ====\n")
    for aa, bb, mm, nn in zip(A, B, M, N):
        print(f"I({aa}, {bb}, {mm}, {nn})")
        testOne(aa, bb, mm, nn)
                