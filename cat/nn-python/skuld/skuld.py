##########################################################################
########################### SKULD NNI LIBRARY ############################
##########################################################################
                  ###          version 0.1          ###
                  ###    a neural network based     ###
                  ### numerical integration library ###
##########################################################################

##########################################################################
###                          LIBRARIES IMPORTS                         ###
##########################################################################
from typing import Tuple, TypeAlias, Annotated, Callable, Any
from torch import Tensor
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from mpmath import polylog
from sklearn.model_selection import train_test_split
import time
import itertools

#########################################################################
###                          EXPLICIT TYPING                          ###
#########################################################################
Vector: TypeAlias = Annotated[Tensor, "torch.float32", (None, 1)]
Matrix: TypeAlias = Annotated[Tensor, "torch.float32", (None, None)]


##########################################################################
###                    MULTILAYERED PERCEPTRON CLASS                  ###
##########################################################################
class MLP(nn.Module):
    """ Class defining neural network approximator for the integrand. """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        """
            Main constructor.

            :param input_size: input layer size (number of integrand dimensions)
            :param hidden_size: hidden layer size (arbitrary value, which would be chosen
                                when training the network)
        """
        super(MLP, self).__init__()
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.input_hidden_layer = nn.Linear(input_size, hidden_size)
        self.sigmoid_activation = nn.Sigmoid()
        self.output_layer = nn.Linear(hidden_size, 1)
        self.criterion: nn.modules.loss = None
        self.optimizer: optim.Optimizer | None = None

    def forward(self, x):
        """
            Forward propagation of the data through the network.

            :param x:    the data to be propagated

            :returns: data after the forward propagation
        """
        x = self.input_hidden_layer(x)  # input->hidden propagation
        x = self.sigmoid_activation(x)  # sigmoid activation function
        x = self.output_layer(x)  # hidden->output propagation

        return x

    def compile(
            self,
            criterion: nn.modules.loss,
            optimizer: optim.Optimizer
    ) -> None:
        """
            Compiles model. Sets learning criterion and model's optimizer to those, provided as params.

            :param criterion: loss function.
            :param optimizer: optimizer for the model.
        """
        self.criterion = criterion
        self.optimizer = optimizer

    def compile_default(self, learning_rate: float) -> None:
        """
            Compiles model with default hyperparams.
            Sets learning criterion to MSE and optimizer to Adam.

            :param learning_rate: rate of the optimizer algorithm.
        """
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def fit(
            self,
            x_train: Matrix,
            y_train: Vector,
            epochs: int,
            verbose: bool = True
    ) -> list[float]:
        """
            Trains the model.

            :param x_train:    Training inputs
            :param y_train:    True labels
            :param epochs:     Number of training epochs
            :param verbose:    Boolean depicting whether should the
            network print each 100 epochs done message, or only the
            completion message.

            :returns: loss functions history
        """
        loss_history: list[float] = []
        start_time: float = time.time()
        for epoch in range(epochs):
            predictions: Vector = self(x_train)  # forward propagation
            loss = self.criterion(predictions, y_train)  # loss function
            self.optimizer.zero_grad()  # gradients are being reset
            loss.backward()  # backwards data propagation
            self.optimizer.step()  # optimization step (network params are being updated)
            loss_history.append(loss.item())  # current loss function added to history
            # hereon the logs are being printed
            if verbose:
                if (epoch + 1) % 100 == 0:
                    print(f'Epoch [{epoch + 1}/{epochs}], '
                          f'Loss: {loss.item():.10f}')
        total_time: float = time.time() - start_time
        print(f'Training done! Time elapsed: {total_time:.2f} seconds')

        return loss_history  # loss history is returned

    def test(self, x_test: Matrix, y_test: Vector) -> float:
        """
            Tests the model.

            :param x_test:    Test inputs
            :param y_test:    True labels

            :returns: test loss function value
        """
        with torch.no_grad():  # no gradients will be calculated
            predictions: Vector = self(x_test)  # forward data propagation
            loss = self.criterion(predictions, y_test)  # loss function

        return loss.item()  # loss function value (item() required to convert tensor to scalar)

    def predict_with(self, x_test: Matrix) -> Vector:
        """
            Uses the model to predict values based on x_test arguments.

            :param x_test: test input

            :returns: predicted function value
        """
        with torch.no_grad():
            prediction: Vector = self(x_test)  # forward data propagation

        return prediction

    def extract_params(self) -> tuple[np.ndarray, np.ndarray,
    np.ndarray, np.ndarray]:
        """
            Extracts weights and biases from the network.

            :returns: tuple of 4 numpy.ndarray-s with biases1, weights1, biases2 and weights2
            (number represents 1 - input->hidden layers params, 2 - hidden->output layers params)
        """
        b1: np.ndarray = self.input_hidden_layer.bias.detach().numpy()
        w1: np.ndarray = self.input_hidden_layer.weight.detach().numpy()
        b2: np.ndarray = self.output_layer.bias.detach().numpy()
        w2: np.ndarray = self.output_layer.weight.detach().numpy().flatten()

        return b1, w1, b2, w2


##########################################################################
###                            MLP WRAPPERS                            ###
##########################################################################
def init_model(
        input_size: int,
        hidden_size: int
) -> MLP:
    return MLP(input_size=input_size, hidden_size=hidden_size)


def split_data(
        X: Matrix,
        y: Vector,
        test_size: float,
        shuffle: bool
) -> Tuple[Matrix, Matrix, Vector, Vector]:
    return train_test_split(X, y, test_size=test_size, shuffle=shuffle)


##########################################################################
###                  NEURAL NETWORK INTEGRATION CLASS                  ###
##########################################################################
class NeuralNumericalIntegration:
    """ Class-toolbox for neural numerical integration routines. """

    @staticmethod
    def calculate2(
            alphas: list[int],
            betas: list[int],
            network_params: tuple[np.ndarray, np.ndarray,
            np.ndarray, np.ndarray]
    ) -> float:
        """
            Calculates 2D integrand value using neural network params
            across given boundaries.

            :param alphas:         lower boundaries sequence (should be placed in integration order)
            :param betas:          upper boundaries sequence (should be placed in integration order)
            :param network_params: params for the trained neural network (default value is 2)

            :returns: integrand value
        """
        alpha1, alpha2, beta1, beta2 = alphas[0], alphas[1], betas[0], betas[1]
        B1, W1, B2, W2 = network_params

        def Phi_j(alp1: float, bt1: float, alp2: float, bt2: float,
                  b1: float, w1_1: float, w1_2: float) -> float:
            term_1: float = polylog(2, -np.exp(-b1 - w1_1 * alp1 - w1_2 * alp2))
            term_2: float = polylog(2, -np.exp(-b1 - w1_1 * alp1 - w1_2 * bt2))
            term_3: float = polylog(2, -np.exp(-b1 - w1_1 * bt1 - w1_2 * alp2))
            term_4: float = polylog(2, -np.exp(-b1 - w1_1 * bt1 - w1_2 * bt2))
            return term_1 - term_2 - term_3 + term_4

        integral_sum: float = 0.0
        for w2_j, w1_1j, w1_2j, b1_j in zip(W2, W1[:, 0], W1[:, 1], B1):
            phi_j: float = Phi_j(alpha1, beta1, alpha2, beta2, b1_j, w1_1j, w1_2j)
            integral_sum += w2_j * ((beta1 - alpha1) * (beta2 - alpha2) + phi_j / (w1_1j * w1_2j))

        return float(B2 * (beta1 - alpha1) * (beta2 - alpha2) + integral_sum)

    @staticmethod
    def calculate1(
            alphas: list[int],
            betas: list[int],
            network_params: tuple[np.ndarray, np.ndarray,
            np.ndarray, np.ndarray]
    ) -> float:
        """
            Calculates 1D integrand value using neural network model params
            across given boundaries.

            :param alphas:         lower boundaries sequence (should be placed in integration order)
            :param betas:          upper boundaries sequence (should be placed in integration order)
            :param network_params: params for the trained neural network (default value is 1)

            :returns: integrand value
        """
        alpha, beta = alphas[0], betas[0]
        B1, W1, B2, W2 = network_params
        W1 = W1.flatten()

        def Phi_j(alp: float, bt: float, b1: float, w1: float) -> float:
            term_alpha: float = polylog(1, -np.exp(-b1 - w1 * alp))
            term_beta: float = polylog(1, -np.exp(-b1 - w1 * bt))
            return term_alpha - term_beta

        integral_sum: float = 0.0
        for w2_j, w1_j, b1_j in zip(W2, W1, B1):
            phi_j: float = Phi_j(alpha, beta, b1_j, w1_j)
            integral_sum += w2_j * ((beta - alpha) + phi_j / w1_j)

        return float(B2 * (beta - alpha) + integral_sum)

    @staticmethod
    def integrate(
            model: MLP,
            alphas: list[int],
            betas: list[int],
            n_dims: int
    ) -> float | None:
        """
            Integrates function-approximator (model) of n-dim dimensions over
            given boundaries.

            :param model: the trained model-approximator
            :param alphas: lower boundaries sequence (should be placed in integration order)
            :param betas:  upper boundaries sequence (should be placed in integration order)
            :param n_dims: number of integrand dimensions

            :returns: neural numeric integration result
        """
        network_params = model.extract_params()
        if n_dims == 1:
            return NeuralNumericalIntegration.calculate1(alphas, betas,
                                                         network_params)
        elif n_dims == 2:
            return NeuralNumericalIntegration.calculate2(alphas, betas,
                                                         network_params)
        else:
            print(f"Integration of functions with {n_dims} dimensions "
                  f"is not implemented.")
            return None


##########################################################################
###                          DATA GENERATORS                           ###
##########################################################################
def generate_data(
        func: Callable[[Matrix, ...], Vector],
        lower: list[float],
        upper: list[float],
        n_samples: int = 100,
        n_dim: int = 1,
        **func_params: Any
) -> tuple[Matrix, Vector]:
    """
        Generates data in the form of a 2D tensor of variables for the function and neural
        network input as well as the function values for the generated tensor of variables.

        :param func:      function to provide values for the variables
        :param lower:     lower bounds of variable values
        :param upper:     upper bounds of variable values
        :param n_samples: number of points of data to generate per dimension (default value 100)
        :param n_dim:     number of dimensions of the function func (default value 1)
        :param *func_args: Additional arguments to pass to the function.

        :returns: dataset of variables X and function values y
    """
    if n_dim == 1:
        X: Matrix = torch.linspace(lower[0], upper[0], n_samples).view(n_samples, 1)
        y: Vector = func(X, **func_params).view(n_samples, 1)
    else:
        ranges = [torch.linspace(lower[n], upper[n], n_samples).tolist()
                  for n in range(n_dim)]
        combinations: list = list(itertools.product(*ranges))
        X: Matrix = torch.tensor(combinations, dtype=torch.float32)
        y: Vector = func(X, **func_params).view(-1, 1)

    return X, y


def generate_data_uniform(
        func: Callable[[Matrix, ...], Vector],
        lower: list[float],
        upper: list[float],
        n_samples: int = 100,
        n_dim: int = 1,
        **func_args: Any
) -> tuple[Matrix, Vector]:
    """
        Generates data UNIFORMLY DISTRIBUTED in the form of a 2D tensor of variables
        for the function and neural network input as well as the function values for
        the generated tensor of variables.

        :param func:      function to provide values for the variables
        :param lower:     lower bounds of variable values
        :param upper:     upper bounds of variable values
        :param n_samples: number of points of data to generate per dimension (default value 100)
        :param n_dim:     number of dimensions of the function func (default value 1)
        :param **func_args: Additional arguments to pass to the function.

        :returns: dataset of variables X and function values y
    """
    if n_dim == 1:
        X: Matrix = torch.rand(n_samples, 1) * (upper[0] - lower[0]) + lower[0]
        y: Vector = func(X, **func_args).view(n_samples, 1)
    else:
        X: Matrix = torch.rand(n_samples, n_dim)
        for i in range(n_dim):
            X[:, i] = X[:, i] * (upper[i] - lower[i]) + lower[i]
        y: Vector = func(X, **func_args).view(n_samples, 1)
    return X, y


##########################################################################
###                           DATA SCALERS                             ###
##########################################################################
def scale_data(
        X_init: Matrix,
        y_init: Vector,
        frange: tuple[int, int] = (0, 1),
) -> tuple[Matrix, Vector]:
    """
        Scales function dataset to the specific range frange.

        :param X_init: initial variables
        :param y_init: initial function values
        :param frange: range of scaled values (default is [0, 1])

        :returns: tuple of scaled X-s and y-s to range frange.
    """
    if not isinstance(X_init, torch.Tensor) or not isinstance(
            y_init, torch.Tensor):
        raise TypeError("Input X_init and y_init must be torch tensors.")
    if X_init.ndim != 2:
        raise ValueError("X_init must be a 2D tensor (m, n_dim).")
    if y_init.ndim != 2 or y_init.shape[1] != 1:
        raise ValueError("y_init must be a 2D tensor (m, 1).")
    if X_init.shape[0] != y_init.shape[0]:
        raise ValueError("X_init and y_init must have the same number "
                         "of rows (m).")
    min_val_x = torch.min(X_init, dim=0).values
    max_val_x = torch.max(X_init, dim=0).values
    min_val_y = torch.min(y_init)
    max_val_y = torch.max(y_init)
    scaled_X = ((frange[1] - frange[0]) * (X_init - min_val_x) /
                (max_val_x - min_val_x) + frange[0])
    scaled_y = ((frange[1] - frange[0]) * (y_init - min_val_y) /
                (max_val_y - min_val_y) + frange[0])
    return scaled_X, scaled_y


def descale_result(
        nni_scaled: float,
        X_init: Matrix,
        y_init: Vector,
        frange: tuple[int, int] = (0, 1),
        n_dim: int = 1
) -> float:
    """
        Restores true value to the scaled integral.

        :param nni_scaled: scaled integral value
        :param X_init:     initial variables
        :param y_init:     initial function values
        :param frange:     range of scaled values (default is [0, 1])
        :param n_dim:      number of function dimensions (default is 1)

        :returns: descaled integral value.
    """
    if not isinstance(X_init, torch.Tensor) or not isinstance(
            y_init, torch.Tensor):
        raise TypeError("Inputs must be torch tensors.")
    if X_init.ndim != 2:
        raise ValueError("X_init must be a 2D tensor.")
    if y_init.ndim != 1 and y_init.ndim != 2:
        raise ValueError("y_init must be a 2D tensor with shape "
                         "(num_of_points,1).")
    if y_init.ndim == 2 and y_init.shape[1] != 1:
        raise ValueError("y_init must be a 2D tensor with shape "
                         "(num_of_points,1).")

    x_min = torch.min(X_init, dim=0).values
    x_max = torch.max(X_init, dim=0).values
    f_min = torch.min(y_init)
    f_max = torch.max(y_init)
    frange_size = frange[1] - frange[0]
    VS = torch.prod(x_max - x_min)
    VSS = frange_size ** n_dim

    return (nni_scaled * (VS * (f_max - f_min) / (VSS * frange_size)) +
            (f_min - (f_max - f_min) / frange_size * frange[0]) * VS).item()
