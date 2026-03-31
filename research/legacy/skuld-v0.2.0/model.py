##########################################################################
###                          LIBRARIES IMPORTS                         ###
##########################################################################
import time
from typing import Tuple, TypeAlias, Annotated, Final

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.nn import Linear, Sigmoid
# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer

#########################################################################
###                          EXPLICIT TYPING                          ###
#########################################################################
Vector: TypeAlias = Annotated[Tensor, "torch.float32", (None, 1)]
Matrix: TypeAlias = Annotated[Tensor, "torch.float32", (None, None)]

#########################################################################
###                              DEVICE                               ###
#########################################################################
DEVICE: torch.device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"SKULD is using device: {DEVICE}")


def get_device(requested_device: str | None = None) -> torch.device:
    selected: str
    device: torch.device

    if requested_device is None:
        selected = (     "cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available()
                    else "cpu")
    else:
        selected = requested_device.lower()
    if selected == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available. Using CPU instead.")
        selected = "cpu"
    if selected == "mps" and not torch.backends.mps.is_available():
        print("WARNING: MPS not available. Using CPU instead.")
        selected = "cpu"
    device = torch.device(selected)
    print(f"SKULD is using device: {device}")

    return device


def set_global_device(name: str) -> None:
    global DEVICE
    DEVICE = get_device(name)

##########################################################################
###                    MULTILAYERED PERCEPTRON CLASS                  ###
##########################################################################
class MLP(nn.Module):
    """ Class defining neural network approximator for the integrand. """

    def __init__(self,
                 input_size: int,
                 hidden_size: int
                 ) -> None:
        """
            Main constructor.

            :param input_size: input layer size (number of integrand dimensions)
            :param hidden_size: hidden layer size (arbitrary value, which would be chosen
                                when training the network)
        """
        super(MLP, self).__init__()
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.input_hidden_layer: Linear = nn.Linear(input_size, hidden_size)
        self.sigmoid_activation: Sigmoid = nn.Sigmoid()
        self.output_layer: Linear = nn.Linear(hidden_size, 1)
        self.criterion: _Loss = None   # type: ignore[annotation-unchecked]
        self.optimizer: Optimizer = None   # type: ignore[annotation-unchecked]
        self.to(DEVICE)


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
            criterion: _Loss,
            optimizer_name: str,
            learning_rate: float = 1e-3,
            **kwargs
        ) -> None:
            """
                Flexible compilation with named optimizer. 
                Supported: 'adam', 'adamw', 'sgd', 'rmsprop', 'lbfgs'

                Args:
                    criterion (_Loss):     loss function.
                    optimizer_name (str):  optimization function name.
                    learning_rate (float): learning rate.
            """
            self.criterion = criterion
            if optimizer_name.lower() == 'adam':
                self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, **kwargs)
            elif optimizer_name.lower() == 'adamw':
                self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, **kwargs)
            elif optimizer_name.lower() == 'sgd':
                self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, **kwargs)
            elif optimizer_name.lower() == 'rmsprop':
                self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate, **kwargs)
            elif optimizer_name.lower() == 'lbfgs':
                defaults = {'lr': 1.0, 'max_iter': 20}
                defaults.update(kwargs)
                self.optimizer = optim.LBFGS(self.parameters(), **defaults)
            else:
                raise ValueError(f"Unknown optimizer: {optimizer_name}")


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
            
            Args:
                (x_train):    Training inputs
                (y_train):    True labels
                (epochs):     Number of training epochs
                (verbose):    Boolean depicting whether should the
                                    network print each 100 epochs done message, 
                                    or only the completion message.
            
            Returns:
                loss functions history
        """

        if self.criterion is None or self.optimizer is None:
            raise RuntimeError("Model must be compiled before fitting!!!")

        self.to(DEVICE)
        x_train, y_train = x_train.to(DEVICE), y_train.to(DEVICE)

        loss_history: list[float] = []
        start_time: float = time.time()

        # Check if optimizer is L-BFGS
        is_lbfgs = isinstance(self.optimizer, optim.LBFGS)

        for epoch in range(epochs):
            if is_lbfgs:
                # Save initial state (after AdamW) as fallback
                best_loss = float('inf')
                best_state = None
                loss_history_local = []

                def closure():
                    self.optimizer.zero_grad()
                    pred = self(x_train)
                    loss = self.criterion(pred, y_train)
                    loss.backward()
                    return loss

                for epoch in range(epochs):
                    try:
                        loss = self.optimizer.step(closure)
                    except Exception as e:
                        print(f"L-BFGS step {epoch+1} failed: {e}")
                        break

                    loss_val = loss.item() if torch.isfinite(loss) else float('nan')

                    if np.isfinite(loss_val) and loss_val < best_loss:
                        best_loss = loss_val
                        best_state = {
                            'epoch': epoch + 1,
                            'loss': loss_val,
                            'state_dict': self.state_dict(),
                            'optimizer_state': self.optimizer.state_dict()
                        }

                    loss_history_local.append(loss_val)

                    if not np.isfinite(loss_val) or (epoch > 0 and loss_val > 10 * loss_history_local[-2]):
                        print(f"Loss exploded ({loss_val:.3e}) at LBFGS epoch {epoch+1}. Stopping.")
                        break

                    if verbose and (epoch + 1) % 10 == 0:
                        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss_val:.10f}')

                if best_state is not None:
                    print(f"Restoring best model from LBFGS epoch {best_state['epoch']} (loss={best_state['loss']:.2e})")
                    self.load_state_dict(best_state['state_dict'])
                    loss_history.extend(loss_history_local[:best_state['epoch']])  # only up to best
                else:
                    print("No valid LBFGS state found. Keeping pre-LBFGS model.")

                return loss_history
            else:
                pred = self(x_train)
                loss = self.criterion(pred, y_train)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_val = loss.item()

            loss_history.append(loss_val)

            if verbose: 
                if is_lbfgs and (epoch + 1) % 10 == 0:
                    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss_val:.10f}')
                if (epoch + 1) % 100 == 0:
                    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss_val:.10f}')

        total_time = time.time() - start_time
        print(f'Training done! Time elapsed: {total_time:.2f} seconds')
        return loss_history


    def test(self, x_test: Matrix, y_test: Vector) -> float:
        """
            Tests the model.

            :param x_test:    Test inputs
            :param y_test:    True labels

            :returns: test loss function value
        """

        if self.criterion is None or self.optimizer is None:
            raise RuntimeError("Model must be compiled before testing!!!")

        self.to(DEVICE)
        x_test, y_test = x_test.to(DEVICE), y_test.to(DEVICE)

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

        if self.criterion is None or self.optimizer is None:
            raise RuntimeError("Model must be compiled before testing!!!")

        self.to(DEVICE)
        x_test = x_test.to(DEVICE)

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
        b1: np.ndarray = self.input_hidden_layer.bias.detach().cpu().numpy()
        w1: np.ndarray = self.input_hidden_layer.weight.detach().cpu().numpy()
        b2: np.ndarray = self.output_layer.bias.detach().cpu().numpy()
        w2: np.ndarray = self.output_layer.weight.detach().cpu().numpy().flatten()

        return b1, w1, b2, w2


##########################################################################
###                            MLP WRAPPERS                            ###
##########################################################################
def init_model(
        input_size: int,
        hidden_size: int
) -> MLP:
    """
        Initializes an MLP model with default hyperparams (Adam optimizer).
        :param input_size: input layer size (number of integrand dimensions)
        :param hidden_size: hidden layer size (arbitrary value, which would be chosen

    """
    return MLP(input_size=input_size, hidden_size=hidden_size)


def split_data(
        X: Matrix,
        y: Vector,
        test_size: float,
        shuffle: bool
) -> Tuple[Matrix, Matrix, Vector, Vector]:
    return train_test_split(X, y, test_size=test_size, shuffle=shuffle)
