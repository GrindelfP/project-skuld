#########################################################################
###                  NNI FOR 2D ITERATED INTEGRAL (I2)                ###
#########################################################################

#########################################################################
###                         LIBRARIES IMPORTS                         ###
#########################################################################
from typing import TypeAlias, Annotated, Any
import torch
from torch import Tensor
from skuld.skuld import generate_data, scale_data, init_model, split_data, NeuralNumericalIntegration, descale_result
from global_region.global_definitions import *
from meson_integrals.functions_i2 import funcI2_wrapper
from i2_plotter import plot_i2

#########################################################################
###                              PARAMETERS                           ###
#########################################################################
A = [0, 1, 0, 1]
B = [0, 0, 1, 1]
M = [2, 2, 2, 2]
N = [3, 3, 3, 3]


#########################################################################
###                         INTEGRATION FUNCTIONS                     ###
#########################################################################
def integrate(a_: float, b_: float, m_: float, n_: float) -> float:
    # 1. Generate dataset
    X_init, y_init = generate_data(
        func=funcI2_wrapper,
        lower=[0.0, 0.0],
        upper=[1.0, 1.0],
        n_samples=300,
        n_dim=2,
        a=a_,
        b=b_,
        m=m_,
        n=n_
    )
    # 2. Normalize (scale) dataset
    X_scaled, y_scaled = scale_data(X_init, y_init)
    # 3. Split scaled dataset into train and test subsets
    x_train, x_test, y_train, y_test = split_data(X_scaled, y_scaled, test_size=0.1, shuffle=True)
    # 4. Initialize neural network
    model = init_model(input_size=2, hidden_size=25)
    # 5. Compile neural network
    model.compile_default(learning_rate=0.001)
    # 6. Train the neural network
    model.fit(x_train, y_train, epochs=5000)
    # 7. Test the neural network
    test_loss = model.test(x_test, y_test)
    print(f"Test Loss: {test_loss:.10f}")
    # 8. Integrate
    nni_scaled = NeuralNumericalIntegration.integrate(model, [0, 0], [1, 1], n_dims=2)
    # 9. Restore scale
    nni_result = descale_result(nni_scaled, X_init, y_init, n_dim=2)

    return nni_result


#########################################################################
###                              PROGRAM                              ###
#########################################################################
if __name__ == "__main__":
    plot_i2()  # plot functions

    integrals: list[float] = []  # integrals values list
    for a_, b_, m_, n_ in zip(A, B, M, N):
        integrals.append(integrate(a_, b_, m_, n_))  # calculation of each integral

    for integral in integrals:
        print("\nI(f) =", integral)  # printing of the integrals
