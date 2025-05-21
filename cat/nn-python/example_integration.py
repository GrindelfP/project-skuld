#########################################################################
###                         LIBRARIES IMPORTS                         ###
#########################################################################
from typing import TypeAlias, Annotated, Any  # typing enhancements
import torch                                  # for torch-based sin/cos
from torch import Tensor                      # for typing
from skuld.skuld import (
generate_data,scale_data,init_model,split_data,NeuralNumericalIntegration,
descale_result)          # required neural network approach functionality
#########################################################################
###                          EXPLICIT TYPING                          ###
#########################################################################
Vector: TypeAlias = Annotated[Tensor, "torch.float32", (None, 1)]
Matrix: TypeAlias = Annotated[Tensor, "torch.float32", (None, None)]
#########################################################################
###                       INTEGRAND DEFINITION                        ###
#########################################################################
def func(XY: Matrix, *args: Any) -> Vector:        # integrand definition
    X: Vector = XY[:, 0]                  # first column, x variable
    Y: Vector = XY[:, 1]                  # second column, y variable
    return torch.cos(X) * torch.sin(Y)    # cos(x)sin(y)
#########################################################################
###                        DATA PREPROCESSING                         ###
#########################################################################
X_init, y_init = generate_data(
    func=func,          # function to integrate (defined above)
    lower=[0.0, 0.0],   # lower dataset boundaries
    upper=[1.0, 1.0],  # upper dataset boundaries
    n_samples=300,      # number of points (square root for n_dim=2)
    n_dim=2             # number of integrand dimensions
)                       # generate dataset
X_scaled, y_scaled = scale_data(
    X_init=X_init,      # variables
    y_init=y_init,      # function values
    frange=(0, 1),      # normalisation range
)                       # normalize (scale) dataset
x_train, x_test, y_train, y_test = split_data(
    X_scaled,           # scaled variables
    y_scaled,           # scaled function values
    test_size=0.1,      # train/test split (90% train data, 10% test)
    shuffle=True        # do shuffle the data to prevent overfitting
)                       # split scaled dataset into train and test subsets
#########################################################################
###                     INTEGRAND APPROXIMATION                       ###
#########################################################################
model = init_model(
    input_size=2,       # size of input layer (n value)
    hidden_size=25      # size of hidden layer (k value)
)                       # initialize neural network
model.compile_default(
    learning_rate=0.1 # learning rate coefficient of the Adam algorithm
)                       # compile the network with default hyperparams
train_history = model.fit(
    x_train=x_train,    # train dataset variables
    y_train=y_train,    # train dataset function values
    epochs=2500,        # number of training epochs (iterations)
    verbose=True        # print the progress messages to std out
)                       # train the neural network
test_loss = model.test(
    x_test=x_test,     # test dataset variables
    y_test=y_test      # test dataset function values
)                      # test the neural network
print(f"Test Loss: {test_loss:.10f}") # print the test result (MAE value)
#########################################################################
###                             INTEGRATION                           ###
#########################################################################
nni_scaled = NeuralNumericalIntegration.integrate(
    model=model,       # neural network
    alphas=[0, 0],     # lower integration boundaries
    betas=[1, 1],      # upper integration boundaries
    n_dims=2           # number of integrand dimensions
)                      # get scaled numerical integral value
#########################################################################
###                     INTEGRAL VALUE DESCALING                      ###
#########################################################################
nni_result = descale_result(
    nni_scaled,        # scaled numerical integral value
    X_init=X_init,     # unscaled variables
    y_init=y_init,     # unscaled function values
    frange=(0, 1),     # normalisation range
    n_dim=2            # number of integrand dimensions
)                      # descale numerical integral value
print("\nI(f) =", nni_result)  # print calculated numerical integral value
