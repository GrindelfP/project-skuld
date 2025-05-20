#########################################################################
###                  NNI FOR 2D ITERATED INTEGRAL (I2)                ###
#########################################################################

#########################################################################
###                         LIBRARIES IMPORTS                         ###
#########################################################################
from typing import TypeAlias, Annotated, Any
import torch
from torch import Tensor
from skuld.skuld import generate_data, scale_data, init_model, split_data, NeuralNumericalIntegration, descale_result, \
    generate_data_uniform
from global_region.global_definitions import *
from meson_integrals.functions_i2 import funcI2_wrapper
from i2_plotter import plot_i2

#########################################################################
###                              PARAMETERS                           ###
#########################################################################
A: list[int] = [0, 1, 0, 1, 0, 1, 0, 1]
B: list[int] = [0, 0, 1, 1, 0, 0, 1, 1]
M: list[int] = [1, 1, 1, 1, 2, 2, 2, 2]
N: list[int] = [2, 2, 2, 2, 3, 3, 3, 3]
x_ranges: list[tuple[float, float]] = [
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0)
]
y_ranges: list[tuple[float, float]] = [
    (0.0, 25.0),
    (0.0, 25.0),
    (0.0, 25.0),
    (0.0, 25.0),
    (0.0, 25.0),
    (0.0, 25.0),
    (0.0, 25.0),
    (0.0, 25.0)
]


#########################################################################
###                         INTEGRATION FUNCTIONS                     ###
#########################################################################
def integrate(
        a_: float,
        b_: float,
        m_: float,
        n_: float,
        xrange: tuple[float, float],
        yrange: tuple[float, float]
) -> float:
    global k, lr, epochs, N_SIZE, DIS_TYPE

    # k: int = 25
    # lr: float = 0.1
    # epochs: int = 5000
    # N_SIZE: int = 10000
    # DIS_TYPE: str = "SUD"

    # 1. Generate dataset
    X_init, y_init = generate_data_uniform(
        func=funcI2_wrapper,
        lower=[xrange[0], yrange[0]],
        upper=[xrange[1], yrange[1]],
        n_samples=N_SIZE,
        n_dim=2,
        a=a_,
        b=b_,
        m=m_,
        n=n_
    )
    # TODO: 2. CLEAR JUPYTER NOTEBOOK

    # 2. Normalize (scale) dataset
    X_scaled, y_scaled = scale_data(X_init, y_init)
    # 3. Split scaled dataset into train and test subsets
    x_train, x_test, y_train, y_test = split_data(X_scaled, y_scaled, test_size=0.1, shuffle=True)
    # 4. Initialize neural network
    model = init_model(input_size=2, hidden_size=k)
    # 5. Compile neural network
    model.compile_default(learning_rate=lr)
    # 6. Train the neural network
    history = model.fit(x_train, y_train, epochs=epochs)
    # 7. Test the neural network
    test_loss = model.test(x_test, y_test)
    print(f"Test Loss: {test_loss:.10f}")
    # 8. Integrate
    nni_scaled = NeuralNumericalIntegration.integrate(model, [0, 0], [1, 1], n_dims=2)
    # 9. Restore scale
    nni_result = descale_result(nni_scaled, X_init, y_init, n_dim=2)

    with open("results", "a") as results_file:
        results_file.write(
            f"I[{a_}, {b_}, {m_}, {n_}] = {nni_result}. lr = {lr}, epochs = {epochs}, k = {k}, "
            f"N = {N_SIZE}, Distribution type: {DIS_TYPE}\n"
        )

    with open("histories", "a") as history_file:
        history_file.write(f"=======================\nlr = {lr}, epochs = {epochs}\nhist = {history}\n")

    return nni_result


def integrate_no_scaling(
        a_: float,
        b_: float,
        m_: float,
        n_: float,
        xrange: tuple[float, float],
        yrange: tuple[float, float]
) -> float:
    global k, lr, epochs, N_SIZE, DIS_TYPE

    # k: int = 35
    # lr: float = 0.1
    # epochs: int = 5000
    # N_SIZE: int = 90000
    # DIS_TYPE: str = "SUD"

    # 1. Generate dataset
    X_init, y_init = generate_data_uniform(
        func=funcI2_wrapper,
        lower=[xrange[0], yrange[0]],
        upper=[xrange[1], yrange[1]],
        n_samples=N_SIZE,
        n_dim=2,
        a=a_,
        b=b_,
        m=m_,
        n=n_
    )

    # 3. Split scaled dataset into train and test subsets
    x_train, x_test, y_train, y_test = split_data(X_init, y_init, test_size=0.1, shuffle=True)
    # 4. Initialize neural network
    model = init_model(input_size=2, hidden_size=k)
    # 5. Compile neural network
    model.compile_default(learning_rate=lr)
    # 6. Train the neural network
    history = model.fit(x_train, y_train, epochs=epochs)
    # 7. Test the neural network
    test_loss = model.test(x_test, y_test)
    print(f"Test Loss: {test_loss:.10f}")
    # 8. Integrate
    nni_result = NeuralNumericalIntegration.integrate(model, [xrange[0], yrange[0]], [xrange[1], yrange[1]], n_dims=2)

    with open("results", "a") as results_file:
        results_file.write(
            f"I[{a_}, {b_}, {m_}, {n_}] = {nni_result}. lr = {lr}, epochs = {epochs}, k = {k}, "
            f"N = {N_SIZE}, Distribution type: {DIS_TYPE}\n"
        )

    with open("histories", "a") as history_file:
        history_file.write(f"=======================\nlr = {lr}, epochs = {epochs}\nhist = {history}\n")

    return nni_result


#########################################################################
###                              PROGRAM                              ###
#########################################################################
if __name__ == "__main__":
    k: int = 25
    lr: float = 0.05
    epochs: int = 2500
    N_SIZE: int = 5000
    DIS_TYPE: str = "SUD"

    # plot_i2(A, B, M, N, x_ranges, y_ranges)  # plot functions

    with open("results", "a") as file:
        file.write("=======================\nTest 27\n=======================\n")

    integrals: list[float] = []  # integrals values list
    # PROBLEM WITH PASSING PARAMETERS
    for a, b, m, n, xr, yr in zip(A, B, M, N, x_ranges, y_ranges):
        integrals.append(integrate(a, b, m, n, xr, yr))  # calculation of each integral

    for integral in integrals:
        print("\nI(f) =", integral)  # printing of the integrals

    with open("results", "a") as file:
        file.write("\n")
